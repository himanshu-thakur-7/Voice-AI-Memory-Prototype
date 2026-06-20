package httpapi

import (
	"encoding/json"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/voiceai/scheduler/internal/config"
	"github.com/voiceai/scheduler/internal/grpcclient"
	"github.com/voiceai/scheduler/internal/scheduler"
	"github.com/voiceai/scheduler/internal/twiml"
)

// safetyNetTTL releases a call's capacity slot even if Twilio never sends a terminal
// status callback (dropped webhook, crash). A real deployment would tune this to the
// max permitted call length.
const safetyNetTTL = 2 * time.Hour

// Server holds the webhook handlers and the registry that maps a live CallSid to the
// scheduler slot it holds, so the slot is released when the call ends.
type Server struct {
	cfg     config.Config
	sched   *scheduler.Scheduler
	backend grpcclient.Client
	pending sync.Map // callSid(string) -> *pendingCall
}

type pendingCall struct {
	res   scheduler.AdmissionResult
	timer *time.Timer
}

func NewServer(cfg config.Config, sched *scheduler.Scheduler, backend grpcclient.Client) *Server {
	return &Server{cfg: cfg, sched: sched, backend: backend}
}

// Routes wires the HTTP surface. /voice and /call-status are signature-guarded.
func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	base := publicBase(s.cfg.MediaWSURL)
	mux.HandleFunc("POST /voice", signatureGuard(s.cfg.TwilioAuthToken, base, s.handleVoice))
	mux.HandleFunc("POST /call-status", signatureGuard(s.cfg.TwilioAuthToken, base, s.handleCallStatus))
	mux.HandleFunc("GET /healthz", s.handleHealthz)
	return mux
}

// handleVoice is the Twilio incoming-call webhook. It admits the call through the
// Fair-Share scheduler, registers context with the backend, and returns TwiML that
// opens the bidirectional media stream.
func (s *Server) handleVoice(w http.ResponseWriter, r *http.Request) {
	_ = r.ParseForm()
	callSid := r.PostFormValue("CallSid")
	from := r.PostFormValue("From")
	to := r.PostFormValue("To")
	// Tenant identifies the fair-share bucket: prefer an explicit ?tenant= on the
	// webhook URL, else the dialed number, else a shared default.
	tenant := firstNonEmpty(r.URL.Query().Get("tenant"), to, "default")
	userID := from // caller's number is a reasonable stable identity for memory isolation

	// 1) Fair-share admission. Backpressure surfaces as a friendly "busy" message.
	res := s.sched.Admit(r.Context(), tenant, callSid, s.cfg.AdmitTimeout)
	if !res.Admitted {
		slog.Info("call rejected by scheduler", "callSid", callSid, "tenant", tenant, "reason", res.Reason)
		writeTwiML(w, mustTwiML(twiml.Busy("All of our lines are busy right now. Please call back shortly.")))
		return
	}

	// 2) Hand context to the backend BEFORE the media socket opens.
	reg, err := s.backend.RegisterCall(r.Context(), grpcclient.CallContext{
		CallSid:       callSid,
		From:          from,
		To:            to,
		TenantID:      tenant,
		UserID:        userID,
		AffectiveHint: r.URL.Query().Get("affective_hint"),
		Engine:        s.cfg.Engine,
		Custom:        map[string]string{"to": to},
	})
	if err != nil || !reg.Accepted {
		slog.Error("backend RegisterCall failed", "callSid", callSid, "err", err, "accepted", reg.Accepted)
		res.Release() // give the slot straight back — the call won't proceed
		writeTwiML(w, mustTwiML(twiml.Reject("backend unavailable")))
		return
	}

	// 3) Remember the slot so /call-status can release it when the call ends.
	s.storePending(callSid, res)

	// 4) Tell Twilio to open the bidirectional stream. Params ride to the Python side
	//    via Twilio's `start` event (start.customParameters).
	wsURL := firstNonEmpty(reg.MediaWSURL, s.cfg.MediaWSURL)
	body := mustTwiML(twiml.ConnectStream(wsURL, map[string]string{
		"callSid": callSid,
		"tenant":  tenant,
		"userId":  userID,
		"engine":  s.cfg.Engine,
	}))
	slog.Info("call admitted", "callSid", callSid, "tenant", tenant, "engine", s.cfg.Engine, "ws", wsURL)
	writeTwiML(w, body)
}

// handleCallStatus is Twilio's status callback. On a terminal status we release the
// capacity slot back to the fair-share pool.
func (s *Server) handleCallStatus(w http.ResponseWriter, r *http.Request) {
	_ = r.ParseForm()
	callSid := r.PostFormValue("CallSid")
	status := r.PostFormValue("CallStatus")
	if isTerminal(status) {
		s.releasePending(callSid)
		slog.Info("call ended; slot released", "callSid", callSid, "status", status)
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) handleHealthz(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(s.sched.Snapshot())
}

// ── slot registry ────────────────────────────────────────────────────────────

func (s *Server) storePending(callSid string, res scheduler.AdmissionResult) {
	pc := &pendingCall{res: res}
	// Safety net: if the terminal status callback never arrives, reclaim the slot.
	pc.timer = time.AfterFunc(safetyNetTTL, func() {
		slog.Warn("releasing slot via safety-net TTL", "callSid", callSid)
		s.releasePending(callSid)
	})
	s.pending.Store(callSid, pc)
}

func (s *Server) releasePending(callSid string) {
	v, ok := s.pending.LoadAndDelete(callSid)
	if !ok {
		return // already released (idempotent across duplicate callbacks)
	}
	pc := v.(*pendingCall)
	pc.timer.Stop()
	pc.res.Release()
}

// ── helpers ──────────────────────────────────────────────────────────────────

func writeTwiML(w http.ResponseWriter, body []byte) {
	w.Header().Set("Content-Type", "text/xml")
	_, _ = w.Write(body)
}

func mustTwiML(body []byte, err error) []byte {
	if err != nil {
		// TwiML marshaling of our own static structs cannot realistically fail; if it
		// somehow does, fall back to a minimal valid document.
		slog.Error("twiml marshal failed", "err", err)
		return []byte(`<?xml version="1.0" encoding="UTF-8"?><Response/>`)
	}
	return body
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}

func isTerminal(status string) bool {
	switch status {
	case "completed", "busy", "failed", "no-answer", "canceled":
		return true
	}
	return false
}

// publicBase derives the https base URL Twilio used to reach us from the configured
// wss media URL (they share a host). Used only for signature reconstruction.
func publicBase(mediaWSURL string) string {
	b := mediaWSURL
	if len(b) > 6 && b[:6] == "wss://" {
		b = "https://" + b[6:]
	} else if len(b) > 5 && b[:5] == "ws://" {
		b = "http://" + b[5:]
	}
	// strip trailing /media path if present
	if i := lastIndexByte(b, '/'); i > len("https://") {
		b = b[:i]
	}
	return b
}

func lastIndexByte(s string, c byte) int {
	for i := len(s) - 1; i >= 0; i-- {
		if s[i] == c {
			return i
		}
	}
	return -1
}

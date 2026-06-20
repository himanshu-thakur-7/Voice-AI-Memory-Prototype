// Package grpcclient is the scheduler's link to the Python backend's Orchestrator
// service. It pushes CallContext to the backend BEFORE Twilio opens the media socket
// so the backend can warm provider connections and run the pre-call memory lookup,
// removing cold-start latency from the first (most-noticed) conversational turn.
//
// ── Why a stub here ───────────────────────────────────────────────────────────
// To keep this service buildable offline with zero codegen, the default client is a
// logging stub. The real implementation is a ~15-line swap once `make proto` has
// generated ./internal/pb — see grpcClientImpl at the bottom of this file (commented).
package grpcclient

import (
	"context"
	"log/slog"
)

// CallContext mirrors orchestrator.v1.CallContext (proto/orchestrator.proto). Kept as
// a plain struct so this package compiles without the generated stubs.
type CallContext struct {
	CallSid       string
	StreamSid     string
	From          string
	To            string
	TenantID      string
	UserID        string
	AffectiveHint string
	Engine        string
	Custom        map[string]string
}

// RegisterResult is the backend's answer to RegisterCall.
type RegisterResult struct {
	MediaWSURL string // wss URL to embed in TwiML; empty => caller falls back to config
	Accepted   bool
	Reason     string
}

// Client is the minimal surface the webhook needs. An interface (not a concrete
// type) keeps the handler testable with a fake and lets us swap stub ↔ real gRPC.
type Client interface {
	RegisterCall(ctx context.Context, cc CallContext) (RegisterResult, error)
	Close() error
}

// New returns the default (stub) client. Swap to dialGRPC(addr) in production.
func New(addr string) Client {
	slog.Info("orchestrator client: using logging stub", "backend", addr,
		"note", "run `make proto` and switch New() to dialGRPC for real gRPC")
	return &stubClient{backendAddr: addr}
}

type stubClient struct{ backendAddr string }

func (s *stubClient) RegisterCall(_ context.Context, cc CallContext) (RegisterResult, error) {
	slog.Info("RegisterCall (stub)",
		"callSid", cc.CallSid, "tenant", cc.TenantID, "engine", cc.Engine, "from", cc.From)
	// Accept and let the webhook use its configured MEDIA_WS_URL.
	return RegisterResult{Accepted: true, MediaWSURL: "", Reason: "stub-accepted"}, nil
}

func (s *stubClient) Close() error { return nil }

// ─────────────────────────────────────────────────────────────────────────────
// PRODUCTION SWAP (uncomment after `make proto` generates ./internal/pb):
//
//	import (
//	    "google.golang.org/grpc"
//	    "google.golang.org/grpc/credentials/insecure"
//	    pb "github.com/voiceai/scheduler/internal/pb"
//	)
//
//	type grpcClientImpl struct {
//	    conn *grpc.ClientConn
//	    api  pb.OrchestratorClient
//	}
//
//	func dialGRPC(addr string) (Client, error) {
//	    conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
//	    if err != nil { return nil, err }
//	    return &grpcClientImpl{conn: conn, api: pb.NewOrchestratorClient(conn)}, nil
//	}
//
//	func (c *grpcClientImpl) RegisterCall(ctx context.Context, cc CallContext) (RegisterResult, error) {
//	    resp, err := c.api.RegisterCall(ctx, &pb.RegisterCallRequest{Context: &pb.CallContext{
//	        CallSid: cc.CallSid, From: cc.From, To: cc.To, TenantId: cc.TenantID,
//	        UserId: cc.UserID, AffectiveHint: cc.AffectiveHint, Engine: cc.Engine, Custom: cc.Custom,
//	    }})
//	    if err != nil { return RegisterResult{}, err }
//	    return RegisterResult{MediaWSURL: resp.MediaWsUrl, Accepted: resp.Accepted, Reason: resp.Reason}, nil
//	}
//
//	func (c *grpcClientImpl) Close() error { return c.conn.Close() }
// ─────────────────────────────────────────────────────────────────────────────

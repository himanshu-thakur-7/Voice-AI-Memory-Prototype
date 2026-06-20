// Package scheduler implements a Fair-Share admission scheduler for inbound calls.
//
// ── The shape of the problem ──────────────────────────────────────────────────
// A telephony front door is a fan-IN: many calls arrive at once, each on its own
// goroutine (one per HTTP webhook request). We must (a) cap total concurrent calls
// so we don't overrun the ML backend, and (b) share that capacity FAIRLY across
// tenants so one noisy tenant can't starve the rest.
//
// ── The Go ideas on display ───────────────────────────────────────────────────
//   • goroutines            — the dispatcher runs in its own goroutine; each webhook
//                             handler is a goroutine that *asks* the dispatcher for a slot.
//   • channels              — all cross-goroutine communication is via channels, never
//                             shared mutable state. The queues are owned solely by the
//                             dispatcher, so they need no mutex (see tokens.go).
//   • select                — the dispatcher multiplexes "new request", "slot freed",
//                             and "shutdown" on one select; handlers select reply-vs-timeout.
//   • buffered chan as semaphore — `slots` is a counting semaphore: its buffer length
//                             is the number of in-flight calls; a full buffer == at capacity.
//   • request/response over channels — each AdmissionRequest carries its OWN reply
//                             channel; this is Go's idiomatic RPC-between-goroutines pattern.
package scheduler

import (
	"context"
	"sync"
	"sync/atomic"
	"time"
)

// AdmissionRequest is what a webhook handler sends to the dispatcher. The embedded
// reply channel is how the dispatcher answers *this specific* caller.
type AdmissionRequest struct {
	Ctx      context.Context // caller's context; canceled if the caller hangs up / times out
	TenantID string
	CallSid  string
	reply    chan AdmissionResult // buffered(1): the dispatcher never blocks answering
}

// AdmissionResult is the dispatcher's answer. When Admitted, Release MUST be called
// exactly once when the call ends, to return the capacity slot to the pool.
type AdmissionResult struct {
	Admitted bool
	Reason   string
	release  func()
}

// Release returns the capacity slot. Safe to call multiple times (idempotent) and
// safe to call on a non-admitted result (no-op).
func (r AdmissionResult) Release() {
	if r.release != nil {
		r.release()
	}
}

// Stats is a point-in-time snapshot for /healthz and metrics.
type Stats struct {
	Active         int   // calls currently holding a slot
	Queued         int   // requests waiting for a slot
	AdmittedTotal  int64 // cumulative admissions
	RejectedTotal  int64 // cumulative rejections (busy/abandoned)
	MaxConcurrent  int
}

// Scheduler is safe for concurrent use by many webhook goroutines.
type Scheduler struct {
	submit  chan *AdmissionRequest // ingress: handlers -> dispatcher
	freed   chan struct{}          // a call ended -> dispatcher should free a slot
	slots   chan struct{}          // counting semaphore; cap == MaxConcurrent

	setW    chan weightUpdate      // tenant weight changes routed through the dispatcher
	statsRq chan chan Stats        // stats requests routed through the dispatcher (no locks)

	maxConcurrent int

	admittedTotal atomic.Int64
	rejectedTotal atomic.Int64
}

type weightUpdate struct {
	tenant string
	weight int
}

// New builds a scheduler that admits at most maxConcurrent simultaneous calls.
func New(maxConcurrent int) *Scheduler {
	if maxConcurrent < 1 {
		maxConcurrent = 1
	}
	return &Scheduler{
		submit:        make(chan *AdmissionRequest),
		freed:         make(chan struct{}, maxConcurrent), // buffered so Release never blocks
		slots:         make(chan struct{}, maxConcurrent),
		setW:          make(chan weightUpdate),
		statsRq:       make(chan chan Stats),
		maxConcurrent: maxConcurrent,
	}
}

// Run drives the dispatcher loop. It blocks until ctx is canceled, so callers
// typically `go sched.Run(ctx)`. Exactly one goroutine ever executes this body,
// which is why the queues it owns need no locking.
func (s *Scheduler) Run(ctx context.Context) {
	queues := newTenantQueues()

	for {
		select {
		case <-ctx.Done():
			// Graceful shutdown: tell everyone still waiting that we're closing.
			s.drain(queues)
			return

		case req := <-s.submit:
			queues.enqueue(req)
			s.dispatch(queues)

		case <-s.freed:
			// A call ended. Free one capacity token, then see who's been waiting.
			select {
			case <-s.slots: // remove one token (decrement the semaphore)
			default: // defensive: never block the dispatcher if accounting drifted
			}
			s.dispatch(queues)

		case w := <-s.setW:
			queues.setWeight(w.tenant, w.weight)

		case reply := <-s.statsRq:
			// Stats are computed here, inside the owner goroutine, so reading the
			// queue depth is race-free without a mutex.
			reply <- Stats{
				Active:        len(s.slots),
				Queued:        queues.depth(),
				AdmittedTotal: s.admittedTotal.Load(),
				RejectedTotal: s.rejectedTotal.Load(),
				MaxConcurrent: s.maxConcurrent,
			}
		}
	}
}

// dispatch grants slots to queued requests until either capacity is exhausted or the
// queues are empty. Runs only on the dispatcher goroutine.
func (s *Scheduler) dispatch(queues *tenantQueues) {
	for queues.hasWaiting() {
		// Try to take a capacity token WITHOUT blocking. A successful send fills one
		// buffer slot; `default` means the buffer is full == we're at capacity.
		select {
		case s.slots <- struct{}{}:
			// acquired a slot
		default:
			return // at capacity; the queued requests stay put until a Release frees one
		}

		req, _ := queues.dequeue()

		// The caller may have hung up or timed out while queued. Don't burn a slot on
		// a ghost — hand the token back and reject.
		if req.Ctx.Err() != nil {
			<-s.slots
			s.rejectedTotal.Add(1)
			req.reply <- AdmissionResult{Admitted: false, Reason: "abandoned before admission"}
			continue
		}

		s.admittedTotal.Add(1)
		req.reply <- AdmissionResult{Admitted: true, release: s.releaseFunc()}
	}
}

// releaseFunc returns an idempotent closure that frees one slot when the call ends.
// sync.Once guarantees a double-Release (e.g. webhook error path + call-status
// callback both firing) only frees a single token.
func (s *Scheduler) releaseFunc() func() {
	var once sync.Once
	return func() {
		once.Do(func() {
			// freed is buffered to maxConcurrent, so this never blocks the caller.
			s.freed <- struct{}{}
		})
	}
}

// drain is called on shutdown: reject everything still queued so no webhook goroutine
// is left blocked forever waiting for a reply.
func (s *Scheduler) drain(queues *tenantQueues) {
	for queues.hasWaiting() {
		req, _ := queues.dequeue()
		s.rejectedTotal.Add(1)
		req.reply <- AdmissionResult{Admitted: false, Reason: "scheduler shutting down"}
	}
}

// Admit asks the scheduler for a capacity slot, waiting up to `timeout`. It is the
// public entry point used by the webhook handler. The returned result's Release()
// must be called when the call ends (we wire that to Twilio's status callback).
func (s *Scheduler) Admit(parent context.Context, tenantID, callSid string, timeout time.Duration) AdmissionResult {
	ctx, cancel := context.WithTimeout(parent, timeout)
	defer cancel()

	req := &AdmissionRequest{
		Ctx:      ctx,
		TenantID: tenantID,
		CallSid:  callSid,
		reply:    make(chan AdmissionResult, 1), // buffered(1) so the dispatcher never blocks
	}

	// Step 1: hand the request to the dispatcher (or give up if even ingress is jammed).
	select {
	case s.submit <- req:
	case <-ctx.Done():
		s.rejectedTotal.Add(1)
		return AdmissionResult{Admitted: false, Reason: "scheduler busy (ingress timeout)"}
	}

	// Step 2: wait for the verdict.
	select {
	case res := <-req.reply:
		return res
	case <-ctx.Done():
		// We stopped waiting, but the dispatcher might still admit us into the buffered
		// reply a moment later. Reap that reply asynchronously so the slot is never leaked.
		go s.reap(req)
		s.rejectedTotal.Add(1)
		return AdmissionResult{Admitted: false, Reason: "scheduler busy (admit timeout)"}
	}
}

// reap consumes the (guaranteed, buffered) reply for an abandoned Admit. Every
// dequeued request gets exactly one reply, so this never blocks forever; if that
// reply turned out to be an admission, we immediately release the slot.
func (s *Scheduler) reap(req *AdmissionRequest) {
	if res := <-req.reply; res.Admitted {
		res.Release()
	}
}

// SetWeight adjusts a tenant's fair-share weight (routed through the dispatcher).
func (s *Scheduler) SetWeight(tenant string, weight int) {
	s.setW <- weightUpdate{tenant: tenant, weight: weight}
}

// Snapshot returns current scheduler stats.
func (s *Scheduler) Snapshot() Stats {
	reply := make(chan Stats, 1)
	s.statsRq <- reply
	return <-reply
}

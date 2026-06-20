package scheduler

import (
	"context"
	"testing"
	"time"
)

// ── Fairness unit tests (deterministic; exercise tokens.go directly) ──────────

func req(tenant, sid string) *AdmissionRequest {
	return &AdmissionRequest{Ctx: context.Background(), TenantID: tenant, CallSid: sid}
}

func drainOrder(q *tenantQueues) []string {
	var out []string
	for {
		r, ok := q.dequeue()
		if !ok {
			return out
		}
		out = append(out, r.CallSid)
	}
}

func TestTenantQueues_RoundRobin(t *testing.T) {
	q := newTenantQueues()
	// A floods with three; B has one. Round-robin must interleave so B is not starved.
	q.enqueue(req("A", "A1"))
	q.enqueue(req("A", "A2"))
	q.enqueue(req("A", "A3"))
	q.enqueue(req("B", "B1"))

	got := drainOrder(q)
	want := []string{"A1", "B1", "A2", "A3"}
	assertSeq(t, got, want)
}

func TestTenantQueues_WeightedBurst(t *testing.T) {
	q := newTenantQueues()
	q.setWeight("A", 2) // A may be served twice before yielding to B
	q.enqueue(req("A", "A1"))
	q.enqueue(req("A", "A2"))
	q.enqueue(req("A", "A3"))
	q.enqueue(req("B", "B1"))

	got := drainOrder(q)
	want := []string{"A1", "A2", "B1", "A3"}
	assertSeq(t, got, want)
}

func assertSeq(t *testing.T, got, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("at %d: got %v want %v", i, got, want)
		}
	}
}

// ── Scheduler integration tests (goroutines + channels) ──────────────────────

func startScheduler(t *testing.T, max int) *Scheduler {
	t.Helper()
	s := New(max)
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go s.Run(ctx)
	return s
}

func TestScheduler_AdmitAndRelease(t *testing.T) {
	s := startScheduler(t, 1)

	r1 := s.Admit(context.Background(), "tenant", "CA1", time.Second)
	if !r1.Admitted {
		t.Fatalf("first call should be admitted: %s", r1.Reason)
	}

	// Capacity is 1 and it's taken — a second admit with a short timeout must be rejected.
	r2 := s.Admit(context.Background(), "tenant", "CA2", 100*time.Millisecond)
	if r2.Admitted {
		t.Fatal("second call should be rejected (at capacity)")
	}

	// Free the slot; now a new admit succeeds.
	r1.Release()
	r3 := s.Admit(context.Background(), "tenant", "CA3", time.Second)
	if !r3.Admitted {
		t.Fatalf("call after Release should be admitted: %s", r3.Reason)
	}
	r3.Release()
}

func TestScheduler_DoubleReleaseIsSafe(t *testing.T) {
	s := startScheduler(t, 1)
	r := s.Admit(context.Background(), "tenant", "CA1", time.Second)
	if !r.Admitted {
		t.Fatal("expected admit")
	}
	r.Release()
	r.Release() // idempotent: must NOT free a second (non-existent) slot

	// If the double-release had leaked a token, capacity accounting would drift.
	r2 := s.Admit(context.Background(), "tenant", "CA2", time.Second)
	if !r2.Admitted {
		t.Fatalf("capacity should be exactly 1 after idempotent release: %s", r2.Reason)
	}
	if got := s.Snapshot().Active; got != 1 {
		t.Fatalf("expected Active=1, got %d", got)
	}
	r2.Release()
}

func TestScheduler_TimedOutAdmitDoesNotLeakSlot(t *testing.T) {
	s := startScheduler(t, 1)

	held := s.Admit(context.Background(), "tenant", "CA1", time.Second)
	if !held.Admitted {
		t.Fatal("expected admit")
	}

	// This one queues behind the held slot, then times out and walks away.
	timedOut := s.Admit(context.Background(), "tenant", "CA2", 50*time.Millisecond)
	if timedOut.Admitted {
		t.Fatal("expected timeout rejection")
	}

	// Releasing the held slot must NOT silently get consumed by the abandoned request.
	held.Release()

	// Give the dispatcher a beat to reconcile the abandoned request, then prove the
	// slot is genuinely free.
	got := s.Admit(context.Background(), "tenant", "CA3", 500*time.Millisecond)
	if !got.Admitted {
		t.Fatalf("slot leaked: a fresh admit should succeed, got %q", got.Reason)
	}
	got.Release()
}

func TestScheduler_StatsSnapshot(t *testing.T) {
	s := startScheduler(t, 2)
	a := s.Admit(context.Background(), "t", "CA1", time.Second)
	b := s.Admit(context.Background(), "t", "CA2", time.Second)
	snap := s.Snapshot()
	if snap.Active != 2 {
		t.Fatalf("Active = %d, want 2", snap.Active)
	}
	if snap.AdmittedTotal < 2 {
		t.Fatalf("AdmittedTotal = %d, want >= 2", snap.AdmittedTotal)
	}
	a.Release()
	b.Release()
}

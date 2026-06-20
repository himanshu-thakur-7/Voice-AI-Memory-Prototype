package scheduler

// tenantQueues implements the *fairness* half of the Fair-Share scheduler: one
// FIFO queue per tenant, plus a weighted round-robin selector that decides whose
// turn it is next. Sharing one global capacity pool FIFO-style would let a single
// tenant that dials 1,000 times in a burst starve everyone else; round-robin across
// per-tenant queues guarantees each active tenant keeps making progress.
//
// ── Go onboarding note ───────────────────────────────────────────────────────
// There is NOT a single mutex in this file, yet it is completely race-free. The
// trick is *ownership*: this struct is touched by exactly one goroutine — the
// scheduler's dispatcher loop. "Don't communicate by sharing memory; share memory
// by communicating." Requests arrive over a channel; the dispatcher is the sole
// owner of the queues. Single-owner state needs no locks.
type tenantQueues struct {
	weights map[string]int                  // per-tenant burst weight (default 1)
	queues  map[string][]*AdmissionRequest  // tenant -> FIFO of pending admissions
	order   []string                        // rotation order of tenants that have items
	cursor  int                             // index into order of the tenant being served
	served  int                             // consecutive admissions granted to order[cursor]
	total   int                             // total queued across all tenants
}

func newTenantQueues() *tenantQueues {
	return &tenantQueues{
		weights: make(map[string]int),
		queues:  make(map[string][]*AdmissionRequest),
	}
}

// setWeight gives a tenant a larger share: weight N lets it be served up to N times
// in a row before the cursor advances to the next tenant. Weight 1 == pure round-robin.
func (q *tenantQueues) setWeight(tenant string, w int) {
	if w < 1 {
		w = 1
	}
	q.weights[tenant] = w
}

func (q *tenantQueues) weight(tenant string) int {
	if w, ok := q.weights[tenant]; ok {
		return w
	}
	return 1
}

func (q *tenantQueues) hasWaiting() bool { return q.total > 0 }

func (q *tenantQueues) depth() int { return q.total }

// enqueue appends a request to its tenant's FIFO. A brand-new tenant is appended to
// the END of the rotation, which never disturbs the current cursor position.
func (q *tenantQueues) enqueue(req *AdmissionRequest) {
	t := req.TenantID
	if _, ok := q.queues[t]; !ok {
		q.queues[t] = nil
		q.order = append(q.order, t)
	}
	q.queues[t] = append(q.queues[t], req)
	q.total++
}

// dequeue pops the next request according to weighted round-robin. The second
// return value is false when nothing is queued.
func (q *tenantQueues) dequeue() (*AdmissionRequest, bool) {
	if q.total == 0 {
		return nil, false
	}
	t := q.order[q.cursor]
	req := q.queues[t][0]
	q.queues[t] = q.queues[t][1:]
	q.total--
	q.served++

	switch {
	case len(q.queues[t]) == 0:
		// Tenant drained — drop it from the rotation. removeCurrent fixes the cursor
		// so it points at the tenant that should be served next.
		q.removeCurrent()
		q.served = 0
	case q.served >= q.weight(t):
		// Tenant used up its burst allowance — advance to the next tenant.
		q.served = 0
		if len(q.order) > 0 {
			q.cursor = (q.cursor + 1) % len(q.order)
		}
	}
	return req, true
}

// removeCurrent deletes the tenant at the cursor from the rotation. After the slice
// delete, whatever tenant followed has shifted INTO the cursor slot — i.e. the cursor
// already names the next tenant to serve, except when we removed the last element and
// must wrap to the front.
func (q *tenantQueues) removeCurrent() {
	i := q.cursor
	delete(q.queues, q.order[i])
	q.order = append(q.order[:i], q.order[i+1:]...)
	if len(q.order) == 0 {
		q.cursor = 0
		return
	}
	if q.cursor >= len(q.order) {
		q.cursor = 0
	}
}

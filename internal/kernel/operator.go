package kernel

import (
	"context"
	"fmt"
	"sync"
	"time"

	"opp299/internal/protocol"
)

type Store interface {
	SaveCheckpoint(ctx context.Context, cp protocol.Checkpoint) error
}

type activeCheckpoint struct {
	cp            *protocol.Checkpoint
	sealed        map[string]bool
	inflight      map[string][]protocol.InFlight
	snapshotState *protocol.State
	baseOutputs   int
}

type Runner struct {
	cfg    protocol.Config
	store  Store
	queues map[string]*BoundedQueue

	mu             sync.Mutex
	state          *protocol.State
	epoch          int64
	lastBarrierID  int64
	acceptedSeq    map[string]int64
	processedSeq   map[string]int64
	barrierSeen    map[string]int64
	active         *activeCheckpoint
	pendingOutputs []protocol.Output
	allOutputs     []protocol.Output
	metrics        protocol.Metrics
	queueTurn      int

	barrierCh chan protocol.Barrier
	closed    map[string]bool
}

func NewRunner(cfg protocol.Config, store Store) (*Runner, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if cfg.MaxConcurrentCheckpoints > 1 {
		return nil, &ClassifiedError{
			Category: CategoryProtocol,
			Err:      fmt.Errorf("%w: this teaching kernel supports one durable snapshot at a time", ErrConcurrentCheckpoint),
		}
	}
	r := &Runner{
		cfg:    cfg,
		store:  store,
		queues: map[string]*BoundedQueue{},
		state:  protocol.NewState(),
		epoch:  1,
		acceptedSeq: map[string]int64{
			protocol.StreamA: 0,
			protocol.StreamB: 0,
		},
		processedSeq: map[string]int64{
			protocol.StreamA: 0,
			protocol.StreamB: 0,
		},
		barrierSeen: map[string]int64{},
		metrics: protocol.Metrics{
			Processed: map[string]int64{
				protocol.StreamA: 0,
				protocol.StreamB: 0,
			},
			MaxBuffer: cfg.BackpressureLimit,
		},
		barrierCh: make(chan protocol.Barrier, 2),
		closed: map[string]bool{
			protocol.StreamA: false,
			protocol.StreamB: false,
		},
	}
	r.queues[protocol.StreamA] = NewBoundedQueue(cfg.BackpressureLimit)
	r.queues[protocol.StreamB] = NewBoundedQueue(cfg.BackpressureLimit)
	return r, nil
}

func (r *Runner) CloseStream(stream string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed[stream] {
		return
	}
	r.closed[stream] = true
	r.queues[stream].Close()
}

func (r *Runner) AbortSources() {
	for _, stream := range streams() {
		r.CloseStream(stream)
	}
}

func streams() []string {
	return []string{protocol.StreamA, protocol.StreamB}
}

func (r *Runner) Run(ctx context.Context) error {
	for {
		if err := ctx.Err(); err != nil {
			return &ClassifiedError{Category: CategoryCancelled, Err: err}
		}
		if err := r.drainOneBarrier(ctx, false); err != nil {
			return err
		}
		processed, err := r.processQueues()
		if err != nil {
			return err
		}
		if processed {
			continue
		}
		if r.finished() {
			return nil
		}
		select {
		case b := <-r.barrierCh:
			if err := r.handleBarrier(b); err != nil {
				return err
			}
		case <-r.queues[protocol.StreamA].Wait():
		case <-r.queues[protocol.StreamB].Wait():
		case <-ctx.Done():
			return &ClassifiedError{Category: CategoryCancelled, Err: ctx.Err()}
		}
	}
}

func (r *Runner) finished() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, stream := range streams() {
		if !r.closed[stream] || r.queues[stream].Len() > 0 {
			return false
		}
	}
	return r.active == nil
}

func (r *Runner) drainOneBarrier(_ context.Context, blocking bool) error {
	if blocking {
		b := <-r.barrierCh
		return r.handleBarrier(b)
	}
	for {
		// len is a deterministic priority check here because Run is the only
		// reader. A non-blocking select would randomly choose default even when
		// a barrier was ready, allowing a fast source's data to starve it.
		if len(r.barrierCh) == 0 {
			return nil
		}
		select {
		case b := <-r.barrierCh:
			if err := r.handleBarrier(b); err != nil {
				return err
			}
		default:
			return nil
		}
	}
}

func (r *Runner) processQueues() (bool, error) {
	streams := streams()
	if r.queueTurn%2 == 1 {
		streams = []string{protocol.StreamB, protocol.StreamA}
	}
	r.queueTurn++
	for _, stream := range streams {
		q := r.queues[stream]
		r.mu.Lock()
		active := r.active
		sealed := active != nil && active.sealed[stream]
		limit := int64(0)
		if sealed && active.cp.Mode == protocol.Aligned {
			limit = active.cp.Positions[stream].BarrierAfter
		} else if active != nil && active.cp.Mode == protocol.Unaligned {
			if _, hasPosition := active.cp.Positions[stream]; hasPosition {
				limit = active.cp.Positions[stream].BarrierAfter
			} else {
				// Before the matching barrier arrives, every unsealed-stream
				// record is named as unincorporated in-flight data.
				r.mu.Unlock()
				continue
			}
		}
		r.mu.Unlock()
		var (
			rec     protocol.Record
			ok      bool
			skipped int
		)
		if sealed {
			rec, ok, skipped = q.takeReadyAfter(limit)
		} else {
			rec, ok, skipped = q.takeReady()
		}
		if !ok {
			continue
		}
		if skipped > 0 && len(r.barrierCh) > 0 {
			r.mu.Lock()
			r.queueTurn++
			r.mu.Unlock()
			return true, nil
		}
		r.mu.Lock()
		r.applyLocked(rec)
		r.mu.Unlock()
		if len(r.barrierCh) > 0 {
			if err := r.drainOneBarrier(context.Background(), false); err != nil {
				return false, err
			}
		}
		return true, nil
	}
	return false, nil
}

func (r *Runner) handleBarrier(b protocol.Barrier) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if b.Stream != protocol.StreamA && b.Stream != protocol.StreamB {
		return &ClassifiedError{Category: CategoryProtocol, Err: fmt.Errorf("%w: stream %q", ErrUnknownStream, b.Stream)}
	}
	if err := r.validateBarrierLocked(b); err != nil {
		return err
	}
	if r.active == nil {
		return r.startCheckpointLocked(b)
	}
	return r.completeCheckpointLocked(b)
}

func (r *Runner) validateBarrierLocked(b protocol.Barrier) error {
	if r.active != nil && r.active.sealed[b.Stream] {
		return &ClassifiedError{
			Category: CategoryProtocol,
			Err:      fmt.Errorf("%w: stream %s barrier %d", ErrDuplicateBarrier, b.Stream, b.ID),
		}
	}
	wantID := r.lastBarrierID + 1
	if r.active != nil {
		wantID = r.active.cp.ID
	}
	if b.OutOfOrder {
		return &ClassifiedError{
			Category: CategoryProtocol,
			Err:      fmt.Errorf("%w: barrier %d is behind the monotonic frontier %d", ErrBarrierOutOfOrder, b.ID, wantID),
		}
	}
	if r.active == nil && b.ID != wantID {
		return &ClassifiedError{Category: CategoryProtocol, Err: fmt.Errorf("%w: expected %d got %d", ErrBarrierGap, wantID, b.ID)}
	}
	if r.active != nil && b.ID != r.active.cp.ID {
		return &ClassifiedError{Category: CategoryProtocol, Err: fmt.Errorf("%w: checkpoint %d got barrier %d", ErrBarrierGap, r.active.cp.ID, b.ID)}
	}
	if b.After < r.processedSeq[b.Stream] {
		return &ClassifiedError{
			Category: CategoryProtocol,
			Err:      fmt.Errorf("%w: stream %s barrier after %d is behind processed seq %d", ErrBarrierGap, b.Stream, b.After, r.processedSeq[b.Stream]),
		}
	}
	return nil
}

func (r *Runner) startCheckpointLocked(b protocol.Barrier) error {
	cp := &protocol.Checkpoint{
		TaskID:    r.cfg.TaskID,
		ID:        b.ID,
		Mode:      r.cfg.Mode,
		Positions: map[string]protocol.Position{},
		Status:    protocol.CheckpointPending,
		StartedAt: time.Now().UTC().Format(time.RFC3339Nano),
	}
	active := &activeCheckpoint{
		cp:          cp,
		sealed:      map[string]bool{b.Stream: true},
		inflight:    map[string][]protocol.InFlight{},
		baseOutputs: len(r.pendingOutputs),
	}
	q := r.queues[b.Stream]
	if r.cfg.Mode == protocol.Aligned {
		prefix := q.TakeUpTo(b.After)
		active.snapshotState = r.state.Clone()
		for _, rec := range prefix {
			out := r.state.Apply(rec)
			out.Epoch = b.ID
			r.pendingOutputs = append(r.pendingOutputs, out)
			r.allOutputs = append(r.allOutputs, out)
			r.markProcessedLocked(rec)
		}
	} else {
		// A barrier is an out-of-band control event. Records before it on the
		// same input still respect FIFO; records already waiting on the other
		// input become the initial unaligned in-flight set.
		prefix := q.TakeUpTo(b.After)
		active.snapshotState = r.state.Clone()
		active.applyPrefixLocked(r, true, prefix, b.ID)
		other := otherStream(b.Stream)
		for _, rec := range r.queues[other].Snapshot() {
			active.inflight[b.Stream] = append(active.inflight[b.Stream], protocol.InFlight{
				Record:         rec,
				ObservedBefore: true,
			})
		}
		// The slice above belongs to the other stream; fix the key without
		// changing record ordering.
		active.inflight[other] = active.inflight[b.Stream]
		delete(active.inflight, b.Stream)
	}
	cp.Positions[b.Stream] = protocol.Position{Stream: b.Stream, Seq: b.After, BarrierAfter: b.After}
	r.barrierSeen[b.Stream] = b.ID
	r.active = active
	return nil
}

func (r *Runner) completeCheckpointLocked(b protocol.Barrier) error {
	active := r.active
	q := r.queues[b.Stream]
	if active.cp.Mode == protocol.Aligned {
		active.applyPrefixLocked(r, true, q.TakeUpTo(b.After), b.ID)
	} else {
		known := map[string]bool{}
		for _, items := range active.inflight {
			for _, item := range items {
				known[item.Record.ID()] = true
			}
		}
		prefix := q.TakeUpTo(b.After)
		for _, rec := range prefix {
			if !known[rec.ID()] {
				active.inflight[b.Stream] = append(active.inflight[b.Stream], protocol.InFlight{Record: rec})
			}
		}
		active.cp.Metrics = r.snapshotMetricsLocked()
		active.applyPrefixLocked(r, false, prefix, b.ID)
	}
	active.sealed[b.Stream] = true
	active.cp.Positions[b.Stream] = protocol.Position{Stream: b.Stream, Seq: b.After, BarrierAfter: b.After}
	r.barrierSeen[b.Stream] = b.ID

	if r.cfg.Mode == protocol.Unaligned {
		bufferOccupancy := len(active.inflight[b.Stream]) + len(active.inflight[otherStream(b.Stream)])
		active.cp.State = active.snapshotState.Clone()
		filtered := map[string][]protocol.InFlight{}
		for stream, items := range active.inflight {
			after := int64(0)
			if pos, ok := active.cp.Positions[stream]; ok {
				after = pos.BarrierAfter
			}
			for _, item := range items {
				if item.Record.Seq <= after {
					filtered[stream] = append(filtered[stream], item)
				}
			}
		}
		active.cp.InFlight = cloneInflight(filtered)
		metrics := r.snapshotMetricsLocked()
		metrics.BufferOccupancy = bufferOccupancy
		active.cp.Metrics = metrics
	} else {
		active.cp.State = r.state.Clone()
		active.cp.InFlight = map[string][]protocol.InFlight{}
	}
	committed := make([]protocol.Output, 0, len(r.pendingOutputs)-active.baseOutputs)
	remaining := r.pendingOutputs[:active.baseOutputs]
	for _, out := range r.pendingOutputs[active.baseOutputs:] {
		if out.Epoch == active.cp.ID {
			committed = append(committed, out)
		} else {
			remaining = append(remaining, out)
		}
	}
	r.pendingOutputs = remaining
	active.cp.Outputs = committed
	if active.cp.Mode == protocol.Aligned {
		active.cp.Metrics = r.snapshotMetricsLocked()
	}
	active.cp.Status = protocol.CheckpointComplete
	active.cp.FinishedAt = time.Now().UTC().Format(time.RFC3339Nano)

	// Ownership moves from the open protocol state to the durable snapshot.
	// Because storage is synchronous, the next monotonic barrier can be
	// accepted only after this commit returns.
	r.active = nil
	r.lastBarrierID = active.cp.ID
	r.epoch = active.cp.ID + 1
	cp := *active.cp
	if r.cfg.FailCheckpointID == cp.ID {
		return &ClassifiedError{Category: CategoryStorage, Err: fmt.Errorf("%w at checkpoint %d", ErrSnapshotFailureInjected, cp.ID)}
	}
	if r.store != nil {
		if err := r.store.SaveCheckpoint(context.Background(), cp); err != nil {
			return Classify(err)
		}
	}
	return nil
}

func cloneInflight(in map[string][]protocol.InFlight) map[string][]protocol.InFlight {
	out := make(map[string][]protocol.InFlight, len(in))
	for stream, items := range in {
		out[stream] = append([]protocol.InFlight(nil), items...)
	}
	return out
}

func (a *activeCheckpoint) applyPrefixLocked(r *Runner, includeInSnapshot bool, prefix []protocol.Record, epoch int64) {
	for _, rec := range prefix {
		var out protocol.Output
		if includeInSnapshot {
			out = a.snapshotState.Apply(rec)
			r.state.Apply(rec)
		} else {
			out = r.state.Apply(rec)
		}
		out.Epoch = epoch
		r.pendingOutputs = append(r.pendingOutputs, out)
		r.allOutputs = append(r.allOutputs, out)
		r.markProcessedLocked(rec)
	}
}

func otherStream(stream string) string {
	if stream == protocol.StreamA {
		return protocol.StreamB
	}
	return protocol.StreamA
}

// RestoreFromCheckpoint rebuilds reference state. The replay service then uses
// cp.InFlight to apply only records that the snapshot marked unincorporated.
func RestoreFromCheckpoint(cfg protocol.Config, store Store, cp protocol.Checkpoint) (*Runner, error) {
	r, err := NewRunner(cfg, store)
	if err != nil {
		return nil, err
	}
	if cp.ID < 1 || cp.Status != protocol.CheckpointComplete {
		return nil, &ClassifiedError{Category: CategoryProtocol, Err: fmt.Errorf("only a complete checkpoint can be restored")}
	}
	r.state = cp.State.Clone()
	r.epoch = cp.ID + 1
	r.lastBarrierID = cp.ID
	for stream, pos := range cp.Positions {
		r.acceptedSeq[stream] = pos.BarrierAfter
		r.processedSeq[stream] = pos.BarrierAfter
		r.barrierSeen[stream] = cp.ID
	}
	return r, nil
}

func (r *Runner) apply(rec protocol.Record) protocol.Output {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.applyLocked(rec)
}

func (r *Runner) applyLocked(rec protocol.Record) protocol.Output {
	epoch := r.epoch
	if r.active != nil {
		epoch = r.active.cp.ID
		if r.active.sealed[rec.Stream] && rec.Seq > r.active.cp.Positions[rec.Stream].BarrierAfter {
			epoch = r.active.cp.ID + 1
		}
	}
	out := r.state.Apply(rec)
	out.Epoch = epoch
	r.pendingOutputs = append(r.pendingOutputs, out)
	r.allOutputs = append(r.allOutputs, out)
	r.metrics.Processed[rec.Stream]++
	r.processedSeq[rec.Stream] = rec.Seq
	return out
}

func (r *Runner) markProcessedLocked(rec protocol.Record) {
	r.metrics.Processed[rec.Stream]++
	r.processedSeq[rec.Stream] = rec.Seq
}

func (r *Runner) StateSnapshot() *protocol.State {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.state.Clone()
}

// ApplyRecoveryRecord incorporates one record named by cp.InFlight. Recovery
// deliberately does not update accepted source positions, so resumed sources
// can later start strictly after BarrierAfter.
func (r *Runner) ApplyRecoveryRecord(rec protocol.Record) protocol.Output {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := r.state.Apply(rec)
	out.Epoch = r.epoch
	r.allOutputs = append(r.allOutputs, out)
	r.metrics.Processed[rec.Stream]++
	r.metrics.Replayed++
	return out
}

func (r *Runner) LastBarrierID() int64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.lastBarrierID
}

// ProcessPending performs one deterministic loop iteration. Tests use it to
// place barriers at exact queue positions without relying on goroutine timing.
func (r *Runner) ProcessPending(ctx context.Context) (bool, error) {
	if err := r.drainOneBarrier(ctx, false); err != nil {
		return false, err
	}
	return r.processQueues()
}

func (r *Runner) PendingOutputCount() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.pendingOutputs)
}

func (r *Runner) Outputs() []protocol.Output {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]protocol.Output(nil), r.allOutputs...)
}

func (r *Runner) PushRecord(ctx context.Context, rec protocol.Record) error {
	q, ok := r.queues[rec.Stream]
	if !ok {
		return &ClassifiedError{Category: CategoryProtocol, Err: fmt.Errorf("%w: %q", ErrUnknownStream, rec.Stream)}
	}
	r.mu.Lock()
	expected := r.acceptedSeq[rec.Stream] + 1
	r.mu.Unlock()
	if rec.Seq != expected {
		return &ClassifiedError{
			Category: CategoryProtocol,
			Err:      fmt.Errorf("%w: stream %s expected seq %d got %d", ErrBarrierGap, rec.Stream, expected, rec.Seq),
		}
	}
	if err := q.Put(ctx, rec); err != nil {
		return &ClassifiedError{Category: CategoryCancelled, Err: err}
	}
	r.mu.Lock()
	r.acceptedSeq[rec.Stream] = rec.Seq
	r.mu.Unlock()
	return nil
}

func (r *Runner) PushBarrier(ctx context.Context, b protocol.Barrier) error {
	select {
	case r.barrierCh <- b:
		return nil
	case <-ctx.Done():
		return &ClassifiedError{Category: CategoryCancelled, Err: ctx.Err()}
	}
}

func (r *Runner) SnapshotMetrics() protocol.Metrics {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.snapshotMetricsLocked()
}

func (r *Runner) snapshotMetricsLocked() protocol.Metrics {
	occupied := 0
	for _, q := range r.queues {
		occupied += q.Len()
	}
	m := r.metrics
	m.Processed = map[string]int64{}
	for stream, value := range r.metrics.Processed {
		m.Processed[stream] = value
	}
	m.BufferOccupancy = occupied
	m.PendingOutputs = len(r.pendingOutputs)
	m.CommittedOutputs = len(r.allOutputs) - len(r.pendingOutputs)
	return m
}

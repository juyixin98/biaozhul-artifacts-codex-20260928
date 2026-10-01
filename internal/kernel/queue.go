package kernel

import (
	"context"
	"sync"

	"opp299/internal/protocol"
)

// BoundedQueue is a FIFO with a hard per-stream capacity. Put blocks when the
// limit is reached, which makes backpressure visible to the synthetic source.
type BoundedQueue struct {
	mu      sync.Mutex
	cond    *sync.Cond
	items   []protocol.Record
	limit   int
	closed  bool
	waiters map[chan struct{}]struct{}
}

func NewBoundedQueue(limit int) *BoundedQueue {
	if limit <= 0 {
		panic("queue limit must be positive")
	}
	q := &BoundedQueue{limit: limit, waiters: map[chan struct{}]struct{}{}}
	q.cond = sync.NewCond(&q.mu)
	return q
}

func (q *BoundedQueue) signal() {
	for ch := range q.waiters {
		close(ch)
		delete(q.waiters, ch)
	}
}

// Wait returns a channel that is closed after any queue mutation. It lets the
// operator's single event loop integrate queue readiness with barrier events.
func (q *BoundedQueue) Wait() <-chan struct{} {
	q.mu.Lock()
	defer q.mu.Unlock()
	ch := make(chan struct{})
	q.waiters[ch] = struct{}{}
	return ch
}

func (q *BoundedQueue) Put(ctx context.Context, r protocol.Record) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	for len(q.items) >= q.limit && !q.closed {
		done := make(chan struct{})
		go func() {
			select {
			case <-ctx.Done():
				q.cond.Broadcast()
			case <-done:
			}
		}()
		q.cond.Wait()
		close(done)
		if ctx.Err() != nil {
			return ctx.Err()
		}
	}
	if q.closed {
		return ErrQueueClosed
	}
	q.items = append(q.items, r)
	q.signal()
	q.cond.Broadcast()
	return nil
}

func (q *BoundedQueue) Take(ctx context.Context) (protocol.Record, bool, error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	for len(q.items) == 0 && !q.closed {
		done := make(chan struct{})
		go func() {
			select {
			case <-ctx.Done():
				q.cond.Broadcast()
			case <-done:
			}
		}()
		q.cond.Wait()
		close(done)
		if ctx.Err() != nil {
			return protocol.Record{}, false, ctx.Err()
		}
	}
	if len(q.items) == 0 {
		return protocol.Record{}, false, nil
	}
	r := q.items[0]
	q.items = q.items[1:]
	q.signal()
	q.cond.Broadcast()
	return r, true, nil
}

func (q *BoundedQueue) takeReady() (protocol.Record, bool, int) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if len(q.items) == 0 {
		return protocol.Record{}, false, 0
	}
	r := q.items[0]
	q.items = q.items[1:]
	q.signal()
	return r, true, 1
}

func (q *BoundedQueue) takeReadyAfter(after int64) (protocol.Record, bool, int) {
	q.mu.Lock()
	defer q.mu.Unlock()
	skipped := 0
	for skipped < len(q.items) && q.items[skipped].Seq <= after {
		skipped++
	}
	if skipped == len(q.items) {
		return protocol.Record{}, false, 0
	}
	r := q.items[skipped]
	q.items = append(q.items[:skipped], q.items[skipped+1:]...)
	q.signal()
	return r, true, 1
}

func (q *BoundedQueue) prepend(r protocol.Record) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.items = append([]protocol.Record{r}, q.items...)
	q.signal()
}

// TakeUpTo removes and returns the ordered prefix with Seq <= after. A later
// record already sitting behind the prefix remains blocked behind the barrier.
func (q *BoundedQueue) TakeUpTo(after int64) []protocol.Record {
	q.mu.Lock()
	defer q.mu.Unlock()
	end := 0
	for end < len(q.items) && q.items[end].Seq <= after {
		end++
	}
	out := append([]protocol.Record(nil), q.items[:end]...)
	q.items = append([]protocol.Record(nil), q.items[end:]...)
	if end > 0 {
		q.signal()
		q.cond.Broadcast()
	}
	return out
}

func (q *BoundedQueue) Drain() []protocol.Record {
	q.mu.Lock()
	defer q.mu.Unlock()
	out := append([]protocol.Record(nil), q.items...)
	q.items = nil
	q.signal()
	q.cond.Broadcast()
	return out
}

func (q *BoundedQueue) Snapshot() []protocol.Record {
	q.mu.Lock()
	defer q.mu.Unlock()
	return append([]protocol.Record(nil), q.items...)
}

func (q *BoundedQueue) SnapshotUpTo(after int64) []protocol.Record {
	q.mu.Lock()
	defer q.mu.Unlock()
	end := 0
	for end < len(q.items) && q.items[end].Seq <= after {
		end++
	}
	return append([]protocol.Record(nil), q.items[:end]...)
}

func (q *BoundedQueue) Len() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.items)
}

func (q *BoundedQueue) Close() {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.closed = true
	q.cond.Broadcast()
	q.signal()
}

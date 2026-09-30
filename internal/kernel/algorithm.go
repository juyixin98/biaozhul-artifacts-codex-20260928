package kernel

import (
	"context"
	"errors"
	"fmt"
	"time"

	"opp297/topk/internal/domain"
)

const MaxKeyScore int64 = 1<<63 - 1

type Reader interface {
	ReadPage(ctx context.Context, req domain.PageRequest) (domain.PageResponse, error)
}

type EventKind string

const (
	EventRequest  EventKind = "PAGE_REQUESTED"
	EventAccepted EventKind = "PAGE_ACCEPTED"
	EventRejected EventKind = "PAGE_REJECTED"
	EventStop     EventKind = "STOP_PROVED"
)

type Event struct {
	Kind       EventKind          `json:"kind"`
	ShardID    int                `json:"shard_id"`
	Round      int                `json:"round"`
	Cursor     *domain.Key        `json:"cursor,omitempty"`
	UpperBound *domain.Key        `json:"upper_bound,omitempty"`
	Items      []domain.Key       `json:"items,omitempty"`
	Threshold  *domain.Key        `json:"threshold,omitempty"`
	Reason     string             `json:"reason"`
	Failure    *domain.ShardError `json:"failure,omitempty"`
	RowsRead   int                `json:"rows_read"`
	At         time.Time          `json:"at"`
}

type ShardStatus struct {
	ID         int         `json:"id"`
	Exhausted  bool        `json:"exhausted"`
	Active     bool        `json:"active"`
	UpperBound *domain.Key `json:"upper_bound,omitempty"`
	Cursor     *domain.Key `json:"cursor,omitempty"`
	Pages      int         `json:"pages"`
	RowsRead   int         `json:"rows_read"`
}

type Result struct {
	Generation domain.GenerationID `json:"generation"`
	K          int                 `json:"k"`
	Exact      bool                `json:"exact"`
	Complete   bool                `json:"complete"`
	Keys       []domain.Key        `json:"keys"`
	Statuses   []ShardStatus       `json:"shard_statuses"`
	Events     []Event             `json:"events"`
	Failure    *domain.ShardError  `json:"failure,omitempty"`
	Rounds     int                 `json:"rounds"`
	TotalRows  int                 `json:"total_rows_read"`
}

type Options struct {
	ShardIDs []int
	K        int
	PageSize int
	Timeout  time.Duration
	// BoundSecret is required for every non-final page. In-process test doubles
	// may supply nil only when they never report HasMore.
	BoundSecret []byte
}

var (
	errGenerationMismatch = errors.New("generation mismatch")
	errProtocol           = errors.New("shard protocol violation")
	errBoundViolation     = errors.New("invalid shard upper bound")
)

// Execute performs an exact global top-k merge. Each participant page uses the
// same score/id ordering and carries a proof for the unread p+1 boundary.
func Execute(ctx context.Context, gen domain.GenerationID, opts Options, reader Reader) (Result, error) {
	if opts.K <= 0 {
		return Result{}, fmt.Errorf("k must be positive")
	}
	if opts.PageSize <= 0 {
		return Result{}, fmt.Errorf("page size must be positive")
	}
	if len(opts.ShardIDs) == 0 {
		return Result{}, fmt.Errorf("at least one shard is required")
	}

	top := newMinHeap(opts.K)
	states := make(map[int]*ShardStatus, len(opts.ShardIDs))
	order := make([]int, 0, len(opts.ShardIDs))
	result := Result{Generation: gen, K: opts.K, Keys: []domain.Key{}, Statuses: []ShardStatus{}, Events: []Event{}}

	for _, id := range opts.ShardIDs {
		if _, exists := states[id]; exists {
			return Result{}, fmt.Errorf("duplicate shard id %d", id)
		}
		states[id] = &ShardStatus{ID: id, Active: true, UpperBound: &domain.Key{Score: MaxKeyScore}}
		order = append(order, id)
	}

	emit := func(ev Event) {
		if ev.At.IsZero() {
			ev.At = time.Now().UTC()
		}
		result.Events = append(result.Events, ev)
	}
	snapshot := func() []ShardStatus {
		out := make([]ShardStatus, 0, len(order))
		for _, id := range order {
			out = append(out, *states[id])
		}
		return out
	}
	fail := func(id int, class domain.FailureClass, err error) Result {
		failure := &domain.ShardError{Shard: id, Class: class, Err: err}
		result.Complete = false
		result.Exact = false
		result.Failure = failure
		result.Statuses = snapshot()
		result.Keys = top.Sorted()
		return result
	}

	for {
		result.Rounds++
		for _, id := range order {
			st := states[id]
			if !st.Active {
				continue
			}
			if top.Len() == opts.K && !st.UpperBound.Better(top.Peek()) {
				st.Active = false
				t := top.Peek()
				emit(Event{Kind: EventStop, ShardID: id, Round: result.Rounds, Cursor: cloneKey(st.Cursor), UpperBound: cloneKey(st.UpperBound), Threshold: &t, Reason: "upper_bound_at_or_below_kth_result", RowsRead: result.TotalRows})
				continue
			}

			reqCtx, cancel := context.WithCancel(ctx)
			if opts.Timeout > 0 {
				reqCtx, cancel = context.WithTimeout(ctx, opts.Timeout)
			}
			req := domain.PageRequest{Generation: gen, ShardID: id, Limit: opts.PageSize, Cursor: cloneKey(st.Cursor)}
			emit(Event{Kind: EventRequest, ShardID: id, Round: result.Rounds, Cursor: cloneKey(st.Cursor), UpperBound: cloneKey(st.UpperBound), Threshold: threshold(top), RowsRead: result.TotalRows})
			page, err := reader.ReadPage(reqCtx, req)
			cancel()
			if err != nil {
				class := domain.FailureProtocol
				if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
					class = domain.FailureTimeout
				}
				emit(Event{Kind: EventRejected, ShardID: id, Round: result.Rounds, Cursor: cloneKey(st.Cursor), UpperBound: cloneKey(st.UpperBound), Threshold: threshold(top), Reason: err.Error(), Failure: &domain.ShardError{Shard: id, Class: class, Err: err}, RowsRead: result.TotalRows})
				return fail(id, class, err), nil
			}
			if verr := validatePage(gen, req, page, opts.BoundSecret); verr != nil {
				class := domain.FailureProtocol
				if errors.Is(verr, errGenerationMismatch) {
					class = domain.FailureGenerationMismatch
				} else if errors.Is(verr, errBoundViolation) {
					class = domain.FailureBoundViolation
				}
				emit(Event{Kind: EventRejected, ShardID: id, Round: result.Rounds, Cursor: cloneKey(st.Cursor), UpperBound: cloneKey(st.UpperBound), Threshold: threshold(top), Reason: verr.Error(), Failure: &domain.ShardError{Shard: id, Class: class, Err: verr}, RowsRead: result.TotalRows})
				return fail(id, class, verr), nil
			}

			keys := make([]domain.Key, 0, len(page.Items))
			for _, item := range page.Items {
				k := item.Key()
				keys = append(keys, k)
				if top.Len() < opts.K {
					top.Push(k)
				} else if k.Better(top.Peek()) {
					top.Pop()
					top.Push(k)
				}
			}
			result.TotalRows += page.RowsRead
			st.RowsRead += page.RowsRead
			st.Pages++
			st.Cursor = cloneKey(page.Cursor)
			st.UpperBound = cloneKey(page.UpperBound)
			st.Exhausted = !page.HasMore
			st.Active = page.HasMore

			reason := "page_accepted"
			if !page.HasMore {
				reason = "shard_exhausted"
			}
			emit(Event{Kind: EventAccepted, ShardID: id, Round: result.Rounds, Cursor: cloneKey(page.Cursor), UpperBound: cloneKey(page.UpperBound), Items: keys, Threshold: threshold(top), Reason: reason, RowsRead: result.TotalRows})
		}

		anyActive := false
		for _, id := range order {
			if states[id].Active {
				anyActive = true
				break
			}
		}
		if !anyActive {
			result.Keys = top.Sorted()
			result.Complete = true
			result.Exact = true
			result.Statuses = snapshot()
			t := threshold(top)
			result.Events = append(result.Events, Event{Kind: EventStop, Round: result.Rounds, Threshold: t, Reason: "no_unseen_candidate_can_beat_kth_result", RowsRead: result.TotalRows})
			result.Events[len(result.Events)-1].At = time.Now().UTC()
			return result, nil
		}
	}
}

func validatePage(expectedGen domain.GenerationID, req domain.PageRequest, page domain.PageResponse, secret []byte) error {
	if page.Generation != expectedGen {
		return fmt.Errorf("%w: requested %s got %s", errGenerationMismatch, expectedGen, page.Generation)
	}
	if page.ShardID != req.ShardID {
		return fmt.Errorf("%w: requested shard %d got %d", errProtocol, req.ShardID, page.ShardID)
	}
	if len(page.Items) > req.Limit {
		return fmt.Errorf("%w: returned %d items for limit %d", errProtocol, len(page.Items), req.Limit)
	}
	if page.RowsRead < len(page.Items) || page.RowsRead > req.Limit+1 {
		return fmt.Errorf("%w: rows_read %d outside (items=%d,limit=%d)", errProtocol, page.RowsRead, len(page.Items), req.Limit)
	}

	prev := req.Cursor
	for i, item := range page.Items {
		k := item.Key()
		if prev != nil && !prev.Better(k) {
			return fmt.Errorf("%w: item %d (%s) is not after cursor %s", errProtocol, i, k, prev)
		}
		prev = &k
	}

	if !page.HasMore {
		if page.UpperBound != nil {
			return fmt.Errorf("%w: exhausted page must omit upper bound", errBoundViolation)
		}
		pageCursor := keyOrNil(page.Items)
		if !sameKey(pageCursor, page.Cursor) {
			return fmt.Errorf("%w: exhausted cursor does not identify last returned item", errBoundViolation)
		}
		return nil
	}

	if len(page.Items) != req.Limit {
		return fmt.Errorf("%w: continuation requires a full %d item page, got %d", errProtocol, req.Limit, len(page.Items))
	}
	if page.RowsRead != req.Limit+1 {
		return fmt.Errorf("%w: continuation requires exactly one probe row, rows_read=%d", errBoundViolation, page.RowsRead)
	}
	if page.Cursor == nil {
		return fmt.Errorf("%w: continuation cursor is missing", errBoundViolation)
	}
	last := page.Items[len(page.Items)-1].Key()
	if !sameKey(&last, page.Cursor) {
		return fmt.Errorf("%w: continuation cursor must equal last returned key", errBoundViolation)
	}
	if page.UpperBound == nil {
		return fmt.Errorf("%w: continuation upper bound is missing", errBoundViolation)
	}
	if !page.Cursor.Better(*page.UpperBound) {
		return fmt.Errorf("%w: upper bound %s must rank below cursor %s", errBoundViolation, page.UpperBound, page.Cursor)
	}
	if !VerifyBound(secret, expectedGen, req.ShardID, *page.Cursor, *page.UpperBound, page.BoundProof) {
		return fmt.Errorf("%w: HMAC for cursor=%s bound=%s", errBoundViolation, page.Cursor, page.UpperBound)
	}
	return nil
}

func threshold(h *minHeap) *domain.Key {
	if h.Len() == 0 {
		return nil
	}
	k := h.Peek()
	return &k
}

func cloneKey(k *domain.Key) *domain.Key {
	if k == nil {
		return nil
	}
	c := *k
	return &c
}

func keyPtr(k domain.Key) *domain.Key { return &k }

func keyOrNil(items []domain.Candidate) *domain.Key {
	if len(items) == 0 {
		return nil
	}
	k := items[len(items)-1].Key()
	return &k
}

func sameKey(a, b *domain.Key) bool {
	if a == nil || b == nil {
		return a == b
	}
	return a.Equal(*b)
}

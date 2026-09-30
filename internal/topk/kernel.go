package topk

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	"topk-demo/internal/protocol"
)

const (
	StatePending   = "pending"
	StateOpen      = "open"
	StatePruned    = "pruned"
	StateExhausted = "exhausted"
	StateFailed    = "failed"
)

type ShardClient interface {
	Bound(ctx context.Context, req protocol.BoundRequest) (*protocol.BoundResponse, error)
	Scan(ctx context.Context, req protocol.ScanRequest) (*protocol.ScanResponse, error)
}

type DiagnosticSink interface {
	Record(ctx context.Context, event protocol.EventRecord) error
}

type NopDiagnostics struct{}

func (NopDiagnostics) Record(context.Context, protocol.EventRecord) error { return nil }

type ShardCallError struct {
	Category string
	Message  string
}

func (e *ShardCallError) Error() string { return e.Category + ": " + e.Message }

type Options struct {
	RunID        string
	Generation   string
	K            int
	PageSize     int
	ShardTimeout time.Duration
	Sink         DiagnosticSink
}

type shardState struct {
	id              string
	state           string
	cursor          string
	upper           *protocol.Candidate
	failureCategory string
	failureMessage  string
	proof           protocol.ShardProof
	firstScan       bool
}

type Engine struct {
	client ShardClient
	seq    int64
}

func NewEngine(client ShardClient) *Engine {
	return &Engine{client: client}
}

func (e *Engine) Execute(ctx context.Context, shardIDs []string, opt Options) (*protocol.QueryResponse, error) {
	if opt.K <= 0 {
		return nil, fmt.Errorf("k must be positive")
	}
	if opt.PageSize <= 0 {
		opt.PageSize = 100
	}
	if opt.ShardTimeout <= 0 {
		opt.ShardTimeout = 2 * time.Second
	}
	if opt.Sink == nil {
		opt.Sink = NopDiagnostics{}
	}
	if strings.TrimSpace(opt.Generation) == "" {
		return nil, fmt.Errorf("generation is required")
	}

	states := make(map[string]*shardState, len(shardIDs))
	for _, id := range shardIDs {
		states[id] = &shardState{id: id, state: StatePending, proof: protocol.ShardProof{ShardID: id, State: StatePending}}
	}
	result := &protocol.QueryResponse{
		RunID:      opt.RunID,
		Generation: opt.Generation,
		K:          opt.K,
		Results:    []protocol.Candidate{},
		Proofs:     make([]protocol.ShardProof, 0, len(shardIDs)),
		Errors:     []protocol.ShardError{},
	}
	result.Metrics.ShardsConfigured = len(shardIDs)
	h := NewTopHeap(opt.K)
	atomic.StoreInt64(&e.seq, 0)

	e.event(ctx, opt, "query_start", nil, h, "")
	e.initialScans(ctx, opt, states, result, h)

	for {
		if done, reason := evaluate(states, h, opt, e, ctx); done {
			return e.finalize(ctx, opt, result, h, states, reason), nil
		}
		target := selectNextShard(states, h)
		if target == nil {
			return e.finalize(ctx, opt, result, h, states, protocol.ReasonIncomplete), nil
		}
		e.scanOne(ctx, opt, target, result, h)
	}
}

func (e *Engine) initialScans(ctx context.Context, opt Options, states map[string]*shardState, result *protocol.QueryResponse, h *TopHeap) {
	type answer struct {
		id   string
		resp *protocol.ScanResponse
		err  error
	}
	answers := make(chan answer, len(states))
	for id := range states {
		states[id].firstScan = true
		go func(id string) {
			callCtx, cancel := context.WithTimeout(ctx, opt.ShardTimeout)
			defer cancel()
			st := states[id]
			resp, err := e.client.Scan(callCtx, protocol.ScanRequest{
				Generation: opt.Generation,
				ShardID:    id,
				Cursor:     st.cursor,
				PageSize:   opt.PageSize,
			})
			answers <- answer{id: id, resp: resp, err: err}
		}(id)
	}

	for range states {
		ans := <-answers
		st := states[ans.id]
		e.event(ctx, opt, "initial_scan_received", st, h, "")
		if ans.err != nil {
			failShard(st, ans.err)
			addError(result, st)
			e.event(ctx, opt, "shard_failed", st, h, st.failureMessage)
			continue
		}
		e.acceptScan(ctx, opt, st, ans.resp, result, h, true)
	}
}

func (e *Engine) scanOne(ctx context.Context, opt Options, st *shardState, result *protocol.QueryResponse, h *TopHeap) {
	callCtx, cancel := context.WithTimeout(ctx, opt.ShardTimeout)
	defer cancel()
	e.event(ctx, opt, "scan_requested", st, h, "")
	resp, err := e.client.Scan(callCtx, protocol.ScanRequest{
		Generation: opt.Generation,
		ShardID:    st.id,
		Cursor:     st.cursor,
		PageSize:   opt.PageSize,
	})
	if err != nil {
		st.proof.StopUpper = cloneCandidate(st.upper)
		failShard(st, err)
		addError(result, st)
		e.event(ctx, opt, "shard_failed", st, h, st.failureMessage)
		return
	}
	e.acceptScan(ctx, opt, st, resp, result, h, false)
}

func (e *Engine) acceptScan(ctx context.Context, opt Options, st *shardState, resp *protocol.ScanResponse, result *protocol.QueryResponse, h *TopHeap, initial bool) {
	if err := validateScanResponse(opt.Generation, st.id, st.cursor, opt.PageSize, st.upper, resp, initial); err != nil {
		if resp != nil && resp.NextUpper != nil {
			st.proof.StopUpper = cloneCandidate(resp.NextUpper)
		} else {
			st.proof.StopUpper = cloneCandidate(st.upper)
		}
		failShard(st, &ShardCallError{Category: protocol.FailureInvalidShard, Message: err.Error()})
		addError(result, st)
		e.event(ctx, opt, "page_rejected", st, h, err.Error())
		return
	}

	st.proof.PagesScanned++
	st.proof.BoundProbes += resp.BoundProbes
	st.proof.StorageRowsRead += resp.ReadRows
	st.proof.CandidatesReturned += int64(len(resp.Items))
	result.Metrics.PagesScanned++
	result.Metrics.BoundProbes += resp.BoundProbes
	result.Metrics.StorageRowsRead += resp.ReadRows
	result.Metrics.CandidatesReturned += int64(len(resp.Items))
	result.Metrics.CandidatesConsidered += int64(len(resp.Items))
	for _, item := range resp.Items {
		h.Add(item)
	}

	st.cursor = resp.NextCursor
	st.upper = cloneCandidate(resp.NextUpper)
	if resp.Exhausted {
		st.state = StateExhausted
		st.upper = nil
		st.proof.CursorExhausted = true
		st.proof.AcceptedAs = "scan_exhausted"
		e.event(ctx, opt, "shard_exhausted", st, h, "scan returned terminal page")
	} else {
		st.state = StateOpen
		st.proof.AcceptedAs = "scan_page_accepted"
		e.event(ctx, opt, "scan_accepted", st, h, "page and next upper accepted")
	}
}

func evaluate(states map[string]*shardState, h *TopHeap, opt Options, e *Engine, ctx context.Context) (bool, string) {
	allResolved := true
	allExhausted := true
	hasFailure := false
	ids := sortedStateIDs(states)
	for _, id := range ids {
		st := states[id]
		if st.state == StateFailed {
			hasFailure = true
			continue
		}
		if st.state == StateExhausted {
			continue
		}
		if st.state == StatePruned {
			allExhausted = false
			continue
		}
		allExhausted = false
		if !canEnter(h, st.upper) {
			st.state = StatePruned
			st.proof.StopUpper = cloneCandidate(st.upper)
			st.proof.RejectedReason = "upper cannot displace current kth candidate"
			e.event(ctx, opt, "shard_pruned", st, h, st.proof.RejectedReason)
			continue
		}
		allResolved = false
	}
	if !allResolved {
		return false, ""
	}
	if hasFailure {
		return true, protocol.ReasonIncomplete
	}
	if allExhausted && h.Len() < opt.K {
		return true, protocol.ReasonExactExhausted
	}
	return true, protocol.ReasonExactThreshold
}

func selectNextShard(states map[string]*shardState, h *TopHeap) *shardState {
	var best *shardState
	for _, st := range states {
		if st.state != StateOpen || st.upper == nil {
			continue
		}
		if !canEnter(h, st.upper) {
			continue
		}
		if best == nil || protocol.Better(*st.upper, *best.upper) {
			best = st
		}
	}
	return best
}

func canEnter(h *TopHeap, upper *protocol.Candidate) bool {
	if upper == nil {
		return false
	}
	if h.Len() < h.K() {
		return true
	}
	threshold := h.Threshold()
	return threshold != nil && protocol.Better(*upper, *threshold)
}

func (e *Engine) finalize(ctx context.Context, opt Options, result *protocol.QueryResponse, h *TopHeap, states map[string]*shardState, reason string) *protocol.QueryResponse {
	result.Results = h.Sorted()
	result.Threshold = h.Threshold()
	result.StopReason = reason
	result.Exact = reason == protocol.ReasonExactThreshold || reason == protocol.ReasonExactExhausted
	result.Complete = result.Exact
	result.Metrics.ShardsSucceeded = 0
	result.Metrics.ShardsFailed = 0
	for _, id := range sortedStateIDs(states) {
		st := states[id]
		proof := st.proof
		proof.State = st.state
		proof.FailureCategory = st.failureCategory
		proof.FailureMessage = st.failureMessage
		if st.state == StatePruned && proof.StopUpper == nil {
			proof.StopUpper = cloneCandidate(st.upper)
		}
		if st.state == StateFailed {
			result.Metrics.ShardsFailed++
		} else {
			result.Metrics.ShardsSucceeded++
		}
		result.Proofs = append(result.Proofs, proof)
	}
	e.event(ctx, opt, "query_stop", nil, h, "stop_reason="+reason)
	return result
}

func applyBoundMetrics(proof *protocol.ShardProof, result *protocol.QueryResponse, readRows, probes int64) {
	proof.BoundProbes += probes
	proof.StorageRowsRead += readRows
	result.Metrics.BoundProbes += probes
	result.Metrics.StorageRowsRead += readRows
}

func failShard(st *shardState, err error) {
	st.state = StateFailed
	category, message := classifyError(err)
	st.failureCategory = category
	st.failureMessage = message
}

func classifyError(err error) (string, string) {
	if err == nil {
		return protocol.FailureInvalidShard, "unknown shard failure"
	}
	var callErr *ShardCallError
	if e, ok := err.(*ShardCallError); ok {
		return e.Category, e.Message
	}
	callErr = nil
	_ = callErr
	if errorsIsTimeout(err) {
		return protocol.FailureTimeout, err.Error()
	}
	return protocol.FailureNetwork, err.Error()
}

func addError(result *protocol.QueryResponse, st *shardState) {
	result.Errors = append(result.Errors, protocol.ShardError{
		ShardID:  st.id,
		Category: st.failureCategory,
		Message:  st.failureMessage,
	})
}

func sortedStateIDs(states map[string]*shardState) []string {
	ids := make([]string, 0, len(states))
	for id := range states {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

func cloneCandidate(c *protocol.Candidate) *protocol.Candidate {
	if c == nil {
		return nil
	}
	v := *c
	return &v
}

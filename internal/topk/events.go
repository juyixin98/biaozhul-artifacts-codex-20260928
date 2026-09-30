package topk

import (
	"context"
	"sync/atomic"

	"topk-demo/internal/protocol"
)

func (e *Engine) event(ctx context.Context, opt Options, eventType string, st *shardState, h *TopHeap, message string) {
	ev := protocol.EventRecord{
		RunID:      opt.RunID,
		Sequence:   atomic.AddInt64(&e.seq, 1),
		EventType:  eventType,
		Generation: opt.Generation,
		HeapSize:   h.Len(),
		Message:    message,
	}
	if eventType == "query_stop" {
		ev.Complete = message == "stop_reason="+protocol.ReasonExactThreshold ||
			message == "stop_reason="+protocol.ReasonExactExhausted
		ev.Exact = ev.Complete
	}
	threshold := h.Threshold()
	if threshold != nil {
		score := threshold.Score
		ev.ThresholdScore = &score
		ev.ThresholdItemID = threshold.ItemID
	}
	if st != nil {
		ev.ShardID = st.id
		if st.upper != nil {
			score := st.upper.Score
			ev.UpperScore = &score
			ev.UpperItemID = st.upper.ItemID
		}
		ev.CursorExhausted = st.state == StateExhausted
		ev.PagesScanned = st.proof.PagesScanned
		ev.BoundProbes = st.proof.BoundProbes
		ev.RowsRead = st.proof.StorageRowsRead
		ev.FailureCategory = st.failureCategory
		ev.Complete = false
		ev.Exact = false
	}
	_ = opt.Sink.Record(ctx, ev)
}

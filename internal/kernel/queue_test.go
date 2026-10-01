package kernel_test

import (
	"context"
	"testing"
	"time"

	"opp299/internal/kernel"
	"opp299/internal/protocol"
)

func TestBoundedQueueEnforcesConfiguredBackpressure(t *testing.T) {
	q := kernel.NewBoundedQueue(2)
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if err := q.Put(ctx, protocol.Record{Stream: protocol.StreamA, Seq: 1}); err != nil {
		t.Fatal(err)
	}
	if err := q.Put(ctx, protocol.Record{Stream: protocol.StreamA, Seq: 2}); err != nil {
		t.Fatal(err)
	}
	third := make(chan error, 1)
	go func() {
		third <- q.Put(context.Background(), protocol.Record{Stream: protocol.StreamA, Seq: 3})
	}()
	select {
	case <-third:
		t.Fatal("third Put completed at capacity")
	case <-time.After(20 * time.Millisecond):
	}

	takeCtx, takeCancel := context.WithTimeout(context.Background(), time.Second)
	defer takeCancel()
	got, ok, err := q.Take(takeCtx)
	if err != nil || !ok || got.Seq != 1 {
		t.Fatalf("Take = %+v ok=%v err=%v", got, ok, err)
	}
	select {
	case err := <-third:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("blocked producer did not resume after one item was consumed")
	}
	if q.Len() != 2 {
		t.Fatalf("queue len = %d, want 2", q.Len())
	}
}

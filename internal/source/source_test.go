package source_test

import (
	"testing"

	"opp299/internal/protocol"
	"opp299/internal/source"
)

func TestLoadSampleAndIndependentOracle(t *testing.T) {
	workload, err := source.LoadWorkload("../../samples/two-speed.json")
	if err != nil {
		t.Fatal(err)
	}
	if workload.TaskID != "demo-two-speed" {
		t.Fatalf("task id = %q", workload.TaskID)
	}
	var a, b int64
	var events int
	for _, stream := range workload.Streams {
		for _, event := range stream.Events {
			events++
			if event.Kind == "record" {
				if stream.ID == protocol.StreamA {
					a += event.Record.Value
				} else {
					b += event.Record.Value
				}
			}
		}
	}
	if events != 15 || a != 23 || b != 115 {
		t.Fatalf("sample oracle events=%d A=%d B=%d, want 15/23/115", events, a, b)
	}
}

func TestWorkloadRejectsBarrierPositionMismatch(t *testing.T) {
	bad := source.Workload{
		TaskID: "bad",
		Streams: []source.StreamSpec{
			{ID: protocol.StreamA, Events: []source.Event{
				{Kind: "record", Record: &protocol.Record{Stream: protocol.StreamA, Seq: 1}},
				{Kind: "barrier", Barrier: &protocol.Barrier{ID: 1, Stream: protocol.StreamA, After: 2}},
			}},
			{ID: protocol.StreamB, Events: []source.Event{
				{Kind: "barrier", Barrier: &protocol.Barrier{ID: 1, Stream: protocol.StreamB, After: 0}},
			}},
		},
	}
	if err := bad.Validate(); err == nil {
		t.Fatal("bad barrier position was accepted")
	}
}

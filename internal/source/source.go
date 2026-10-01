package source

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"time"

	"opp299/internal/protocol"
)

type Event struct {
	Kind        string            `json:"kind"`
	DelayMillis int               `json:"delay_millis,omitempty"`
	AfterMillis int               `json:"after_millis,omitempty"`
	DurationMS  int               `json:"duration_millis,omitempty"`
	Record      *protocol.Record  `json:"record,omitempty"`
	Barrier     *protocol.Barrier `json:"barrier,omitempty"`
}

type StreamSpec struct {
	ID           string  `json:"id"`
	DelayMillis  int     `json:"delay_millis"`
	StartDelayMS int     `json:"start_delay_millis,omitempty"`
	Events       []Event `json:"events"`
}

type Workload struct {
	TaskID  string       `json:"task_id"`
	Streams []StreamSpec `json:"streams"`
}

func LoadWorkload(path string) (Workload, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Workload{}, err
	}
	var w Workload
	if err := json.Unmarshal(data, &w); err != nil {
		return Workload{}, err
	}
	if err := w.Validate(); err != nil {
		return Workload{}, err
	}
	return w, nil
}

func (w Workload) Validate() error {
	if w.TaskID == "" {
		return fmt.Errorf("workload task_id is required")
	}
	seen := map[string]bool{}
	for _, stream := range w.Streams {
		if stream.ID != protocol.StreamA && stream.ID != protocol.StreamB {
			return fmt.Errorf("stream must be A or B, got %q", stream.ID)
		}
		if seen[stream.ID] {
			return fmt.Errorf("duplicate stream %q", stream.ID)
		}
		seen[stream.ID] = true
		lastSeq := int64(0)
		lastBarrier := int64(0)
		for _, event := range stream.Events {
			switch event.Kind {
			case "record":
				if event.Record == nil {
					return fmt.Errorf("stream %s has record event without record", stream.ID)
				}
				if event.Record.Stream != stream.ID {
					return fmt.Errorf("record stream %q does not match channel %q", event.Record.Stream, stream.ID)
				}
				if event.Record.Seq != lastSeq+1 {
					return fmt.Errorf("stream %s expected seq %d got %d", stream.ID, lastSeq+1, event.Record.Seq)
				}
				lastSeq = event.Record.Seq
			case "barrier":
				if event.Barrier == nil {
					return fmt.Errorf("stream %s has barrier event without barrier", stream.ID)
				}
				if event.Barrier.Stream != stream.ID {
					return fmt.Errorf("barrier stream %q does not match channel %q", event.Barrier.Stream, stream.ID)
				}
				if event.Barrier.After != lastSeq {
					return fmt.Errorf("stream %s barrier %d after=%d but accepted seq=%d", stream.ID, event.Barrier.ID, event.Barrier.After, lastSeq)
				}
				if event.Barrier.ID <= lastBarrier {
					return fmt.Errorf("workload barrier ids must be monotonic per stream")
				}
				lastBarrier = event.Barrier.ID
			case "pause":
				if event.DurationMS <= 0 {
					return fmt.Errorf("stream %s pause requires positive duration_millis", stream.ID)
				}
			default:
				return fmt.Errorf("unknown event kind %q", event.Kind)
			}
		}
	}
	if !seen[protocol.StreamA] || !seen[protocol.StreamB] {
		return fmt.Errorf("workload requires both streams A and B")
	}
	return nil
}

type Pusher interface {
	PushRecord(context.Context, protocol.Record) error
	PushBarrier(context.Context, protocol.Barrier) error
}

// Run sends one synthetic stream in order. When skipResumeBarrier is true, it
// skips events through the checkpoint barrier; otherwise it sends that barrier
// to the already restored runner. InFlight records are replayed separately.
func Run(ctx context.Context, spec StreamSpec, pusher Pusher, resumeAfter int64, skipResumeBarrier bool) error {
	skipping := resumeAfter > 0
	if !skipping && spec.StartDelayMS > 0 {
		select {
		case <-time.After(time.Duration(spec.StartDelayMS) * time.Millisecond):
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	for _, event := range spec.Events {
		delay := spec.DelayMillis
		if event.DelayMillis > 0 {
			delay = event.DelayMillis
		}
		if delay > 0 && !(skipping && skipResumeBarrier) {
			select {
			case <-time.After(time.Duration(delay) * time.Millisecond):
			case <-ctx.Done():
				return ctx.Err()
			}
		}
		if skipping {
			if event.Kind == "barrier" && event.Barrier != nil && event.Barrier.ID == resumeAfter {
				skipping = false
				if skipResumeBarrier {
					continue
				}
			}
			continue
		}
		switch event.Kind {
		case "record":
			if err := pusher.PushRecord(ctx, *event.Record); err != nil {
				return err
			}
		case "barrier":
			if err := pusher.PushBarrier(ctx, *event.Barrier); err != nil {
				return err
			}
		case "pause":
			select {
			case <-time.After(time.Duration(event.DurationMS) * time.Millisecond):
			case <-ctx.Done():
				return ctx.Err()
			}
		}
		if event.AfterMillis > 0 && !(resumeAfter > 0 && skipResumeBarrier) {
			select {
			case <-time.After(time.Duration(event.AfterMillis) * time.Millisecond):
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	}
	return nil
}

// InflightOrder returns records in replay order. A then B keeps teaching output
// deterministic without changing the reference sums.
func InflightOrder(cp protocol.Checkpoint) []protocol.Record {
	out := []protocol.Record{}
	for _, stream := range []string{protocol.StreamA, protocol.StreamB} {
		items := append([]protocol.InFlight(nil), cp.InFlight[stream]...)
		sort.SliceStable(items, func(i, j int) bool {
			return items[i].Record.Seq < items[j].Record.Seq
		})
		for _, item := range items {
			out = append(out, item.Record)
		}
	}
	return out
}

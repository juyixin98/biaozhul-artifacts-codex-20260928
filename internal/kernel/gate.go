package kernel

import (
	"context"
	"strconv"
	"sync/atomic"

	"checkpointlab/internal/protocol"
)

// Gate is the bounded local transport for one input partition. In aligned mode
// pausing the gate is the barrier-channel block; the channel capacity is the
// hard backpressure upper bound even before a barrier arrives.
type Gate struct {
	stream    string
	capacity  int
	events    chan protocol.Event
	paused    atomic.Bool
	highWater int
	blocked   int64
	accepted  int64
}

func NewGate(stream string, capacity int) *Gate {
	if capacity < 1 {
		capacity = 1
	}
	return &Gate{
		stream:   stream,
		capacity: capacity,
		events:   make(chan protocol.Event, capacity),
	}
}

func (g *Gate) Send(ctx context.Context, event protocol.Event) error {
	if g.paused.Load() {
		atomic.AddInt64(&g.blocked, 1)
		return protocol.NewTypedError(protocol.CategoryAlignmentBlocked,
			"aligned checkpoint has closed gate "+g.stream, true)
	}
	select {
	case g.events <- event:
		atomic.AddInt64(&g.accepted, 1)
		if n := len(g.events); n > g.highWater {
			g.highWater = n
		}
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (g *Gate) TrySend(event protocol.Event) error {
	if g.paused.Load() {
		atomic.AddInt64(&g.blocked, 1)
		return protocol.NewTypedError(protocol.CategoryAlignmentBlocked,
			"aligned checkpoint has closed gate "+g.stream, true)
	}
	select {
	case g.events <- event:
		atomic.AddInt64(&g.accepted, 1)
		if n := len(g.events); n > g.highWater {
			g.highWater = n
		}
		return nil
	default:
		return protocol.NewTypedError(protocol.CategoryBackpressureLimit,
			"gate "+g.stream+" reached capacity "+strconv.Itoa(g.capacity), false)
	}
}

func (g *Gate) Events() <-chan protocol.Event { return g.events }
func (g *Gate) Pause() bool                   { return g.paused.CompareAndSwap(false, true) }
func (g *Gate) Resume()                       { g.paused.Store(false) }
func (g *Gate) Paused() bool                  { return g.paused.Load() }
func (g *Gate) Close()                        { close(g.events) }

type GateStats struct {
	Stream          string `json:"stream"`
	Capacity        int    `json:"capacity"`
	Occupancy       int    `json:"occupancy"`
	HighWaterMark   int    `json:"high_water_mark"`
	Paused          bool   `json:"paused"`
	BlockedAttempts int64  `json:"blocked_attempts"`
	Accepted        int64  `json:"accepted"`
}

func (g *Gate) Stats() GateStats {
	return GateStats{
		Stream:          g.stream,
		Capacity:        g.capacity,
		Occupancy:       len(g.events),
		HighWaterMark:   g.highWater,
		Paused:          g.paused.Load(),
		BlockedAttempts: atomic.LoadInt64(&g.blocked),
		Accepted:        atomic.LoadInt64(&g.accepted),
	}
}

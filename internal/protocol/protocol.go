package protocol

import "fmt"

const (
	StreamA = "A"
	StreamB = "B"
)

type Mode string

const (
	ModeAligned   Mode = "aligned"
	ModeUnaligned Mode = "unaligned"
)

func (m Mode) Valid() bool {
	return m == ModeAligned || m == ModeUnaligned
}

type EventKind string

const (
	KindRecord  EventKind = "record"
	KindBarrier EventKind = "barrier"
)

// Record is one synthetic keyed update from one of the two input partitions.
// Offset is zero based and contiguous per stream. Barrier.Offset is exclusive.
type Record struct {
	Stream string `json:"stream"`
	Offset int64  `json:"offset"`
	Epoch  int64  `json:"epoch"`
	Key    string `json:"key"`
	Value  int64  `json:"value"`
}

type Barrier struct {
	Stream string `json:"stream"`
	Epoch  int64  `json:"epoch"`
	Offset int64  `json:"offset"`
}

type Event struct {
	Kind    EventKind `json:"kind"`
	Record  Record    `json:"record,omitempty"`
	Barrier Barrier   `json:"barrier,omitempty"`
}

func RecordEvent(r Record) Event {
	return Event{Kind: KindRecord, Record: r}
}

func BarrierEvent(b Barrier) Event {
	return Event{Kind: KindBarrier, Barrier: b}
}

func ValidStream(stream string) bool {
	return stream == StreamA || stream == StreamB
}

func (r Record) Validate() error {
	if !ValidStream(r.Stream) {
		return fmt.Errorf("unknown stream %q", r.Stream)
	}
	if r.Offset < 0 {
		return fmt.Errorf("record offset must be non-negative")
	}
	if r.Epoch < 1 {
		return fmt.Errorf("record epoch must be positive")
	}
	if r.Key == "" {
		return fmt.Errorf("record key is required")
	}
	return nil
}

func (b Barrier) Validate() error {
	if !ValidStream(b.Stream) {
		return fmt.Errorf("unknown stream %q", b.Stream)
	}
	if b.Epoch < 1 {
		return fmt.Errorf("barrier epoch must be positive")
	}
	if b.Offset < 0 {
		return fmt.Errorf("barrier offset must be non-negative")
	}
	return nil
}

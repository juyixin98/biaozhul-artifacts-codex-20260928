package kernel

import "errors"

var (
	ErrQueueClosed             = errors.New("queue closed")
	ErrBarrierOutOfOrder       = errors.New("barrier id is not monotonic")
	ErrBarrierGap              = errors.New("barrier sequence has a gap")
	ErrDuplicateBarrier        = errors.New("duplicate barrier")
	ErrConcurrentCheckpoint    = errors.New("too many concurrent checkpoints")
	ErrUnknownStream           = errors.New("unknown stream")
	ErrCheckpointNotFound      = errors.New("checkpoint not found")
	ErrSnapshotFailureInjected = errors.New("snapshot failure injected")
)

type ErrorCategory string

const (
	CategoryProtocol  ErrorCategory = "protocol_violation"
	CategoryCapacity  ErrorCategory = "capacity_exceeded"
	CategoryStorage   ErrorCategory = "storage_failure"
	CategoryCancelled ErrorCategory = "cancelled"
	CategoryNotFound  ErrorCategory = "not_found"
)

type ClassifiedError struct {
	Category ErrorCategory
	Err      error
}

func (e *ClassifiedError) Error() string { return e.Err.Error() }
func (e *ClassifiedError) Unwrap() error { return e.Err }

func Classify(err error) *ClassifiedError {
	if err == nil {
		return nil
	}
	var classified *ClassifiedError
	if errors.As(err, &classified) {
		return classified
	}
	return &ClassifiedError{Category: CategoryStorage, Err: err}
}

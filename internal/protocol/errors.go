package protocol

type ErrorCategory string

const (
	CategoryValidation        ErrorCategory = "VALIDATION_REJECTED"
	CategoryBarrierOrder      ErrorCategory = "BARRIER_ORDER_VIOLATION"
	CategoryCheckpointScope   ErrorCategory = "CONCURRENT_CHECKPOINT_REJECTED"
	CategoryAlignmentBlocked  ErrorCategory = "ALIGNED_INPUT_BLOCKED"
	CategoryReplayGap         ErrorCategory = "REPLAY_OFFSET_GAP"
	CategoryReplayDuplicate   ErrorCategory = "REPLAY_DUPLICATE_OFFSET"
	CategoryStorageFailed     ErrorCategory = "STORAGE_WRITE_FAILED"
	CategoryUncertainStorage  ErrorCategory = "UNCERTAIN_STORAGE_OUTCOME"
	CategoryBackpressureLimit ErrorCategory = "BACKPRESSURE_LIMIT_REACHED"
)

type TypedError struct {
	Category  ErrorCategory `json:"category"`
	Message   string        `json:"message"`
	Retriable bool          `json:"retriable"`
}

func (e *TypedError) Error() string {
	return string(e.Category) + ": " + e.Message
}

func NewTypedError(category ErrorCategory, message string, retriable bool) *TypedError {
	return &TypedError{Category: category, Message: message, Retriable: retriable}
}

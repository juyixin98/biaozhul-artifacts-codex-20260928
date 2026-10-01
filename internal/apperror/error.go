package apperror

import "fmt"

// Kind is the stable error contract used by HTTP handlers and tests.
type Kind string

const (
	KindInvalidInput      Kind = "INVALID_INPUT"
	KindStateConflict     Kind = "STATE_CONFLICT"
	KindResourceExhausted Kind = "RESOURCE_EXHAUSTED"
	KindComputationFailed Kind = "COMPUTATION_FAILED"
	KindNotFound          Kind = "NOT_FOUND"
	KindUnavailable       Kind = "UNAVAILABLE"
)

type Error struct {
	Kind    Kind
	Code    string
	Message string
	Cause   error
}

func (e *Error) Error() string {
	if e.Cause == nil {
		return fmt.Sprintf("%s: %s", e.Code, e.Message)
	}
	return fmt.Sprintf("%s: %s: %v", e.Code, e.Message, e.Cause)
}

func (e *Error) Unwrap() error { return e.Cause }

func New(kind Kind, code, message string) *Error {
	return &Error{Kind: kind, Code: code, Message: message}
}

func Wrap(kind Kind, code, message string, cause error) *Error {
	return &Error{Kind: kind, Code: code, Message: message, Cause: cause}
}

func InvalidInput(code, message string) *Error {
	return New(KindInvalidInput, code, message)
}

func Conflict(code, message string) *Error {
	return New(KindStateConflict, code, message)
}

func ResourceExhausted(code, message string) *Error {
	return New(KindResourceExhausted, code, message)
}

func ComputationFailed(code, message string) *Error {
	return New(KindComputationFailed, code, message)
}

func NotFound(code, message string) *Error {
	return New(KindNotFound, code, message)
}

func Unavailable(code, message string, cause error) *Error {
	return Wrap(KindUnavailable, code, message, cause)
}

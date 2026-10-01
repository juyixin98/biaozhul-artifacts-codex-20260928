package protocol

import "errors"

type ErrorClass string

const (
	ClassInput             ErrorClass = "input_error"
	ClassState             ErrorClass = "state_conflict"
	ClassNotFound          ErrorClass = "not_found"
	ClassResourceExhausted ErrorClass = "resource_exhausted"
	ClassComputation       ErrorClass = "computation_failure"
	ClassStorage           ErrorClass = "storage_failure"
)

type ServiceError struct {
	Class   ErrorClass
	Code    string
	Message string
	Cause   error
}

func (e *ServiceError) Error() string {
	if e == nil {
		return ""
	}
	return string(e.Class) + ": " + e.Message
}

func (e *ServiceError) Unwrap() error { return e.Cause }

func NewServiceError(class ErrorClass, code, message string, cause error) *ServiceError {
	return &ServiceError{Class: class, Code: code, Message: message, Cause: cause}
}

func AsServiceError(err error) (*ServiceError, bool) {
	var serviceErr *ServiceError
	if errors.As(err, &serviceErr) {
		return serviceErr, true
	}
	return nil, false
}

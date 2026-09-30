package topk

import (
	"context"
	"errors"

	"topk-demo/internal/protocol"
)

func errorsIsTimeout(err error) bool {
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	type timeoutError interface{ Timeout() bool }
	var te timeoutError
	if errors.As(err, &te) {
		return te.Timeout()
	}
	return false
}

func failureForHTTPStatus(status int) string {
	if status == 409 {
		return protocol.FailureBadGeneration
	}
	if status >= 400 && status < 600 {
		return protocol.FailureHTTPStatus
	}
	return protocol.FailureNetwork
}

package worker

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"time"

	"opp300-speculative-batch/internal/participant"
	"opp300-speculative-batch/internal/protocol"
)

type Kernel struct {
	Client *participant.Client
	Now    func() time.Time
}

type Result struct {
	Output         string
	ChecksumSHA256 string
	SizeBytes      int64
}

func (k *Kernel) Compute(input string, behavior protocol.Behavior) (Result, error) {
	if strings.TrimSpace(input) == "" {
		return Result{}, &FailureError{Category: protocol.FailureValidation, Message: "input is empty"}
	}
	if behavior.Fail {
		category := behavior.FailureCategory
		if category == "" || !knownFailureCategory(category) {
			category = protocol.FailureCompute
		}
		message := behavior.FailureMessage
		if message == "" {
			message = "synthetic worker failure"
		}
		return Result{}, &FailureError{Category: category, Message: message}
	}
	lines := strings.Split(input, "\n")
	for i, line := range lines {
		lines[i] = strings.ToUpper(line)
	}
	output := strings.Join(lines, "\n") + "\n"
	sum := sha256.Sum256([]byte(output))
	return Result{
		Output:         output,
		ChecksumSHA256: hex.EncodeToString(sum[:]),
		SizeBytes:      int64(len(output)),
	}, nil
}

func knownFailureCategory(category string) bool {
	switch category {
	case protocol.FailureValidation, protocol.FailureCompute, protocol.FailureWorker,
		protocol.FailureLeaseExpired, protocol.FailureInterrupted, protocol.FailureSuperseded:
		return true
	default:
		return false
	}
}

func (k *Kernel) Run(ctx context.Context, req participant.ExecuteRequest) error {
	if req.AttemptID == "" || req.CallbackBaseURL == "" {
		return &FailureError{Category: protocol.FailureValidation, Message: "attempt_id and callback_base_url are required"}
	}
	leaseFor := time.Duration(req.LeaseMS) * time.Millisecond
	if leaseFor <= 0 {
		leaseFor = 5 * time.Second
	}
	heartbeatInterval := time.Duration(req.HeartbeatIntervalMS) * time.Millisecond
	if heartbeatInterval <= 0 {
		heartbeatInterval = 500 * time.Millisecond
	}

	startCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	err := k.Client.Start(startCtx, req.CallbackBaseURL, req.AttemptID, participant.StartRequest{
		Epoch:     req.Epoch,
		WorkerURL: ServerURL(ctx),
	})
	if err != nil {
		return err
	}

	fail := func(category, message string) error {
		failCtx, failCancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer failCancel()
		if err := k.Client.Fail(failCtx, req.CallbackBaseURL, req.AttemptID, participant.FailureRequest{
			Epoch:           req.Epoch,
			FailureCategory: category,
			FailureMessage:  message,
			WorkerURL:       ServerURL(ctx),
		}); err != nil {
			message = fmt.Sprintf("%s; callback error: %v", message, err)
		}
		return &FailureError{Category: category, Message: message}
	}

	heartbeatCtx, stopHeartbeats := context.WithCancel(ctx)
	defer stopHeartbeats()
	heartbeatDone := make(chan struct{})
	go k.heartbeatLoop(heartbeatCtx, heartbeatDone, req, heartbeatInterval, leaseFor)

	delayMS := effectiveDelay(req)
	if delayMS > 0 {
		select {
		case <-time.After(time.Duration(delayMS) * time.Millisecond):
		case <-ctx.Done():
			stopHeartbeats()
			<-heartbeatDone
			return ctx.Err()
		}
	}

	result, err := k.Compute(req.Input, req.Behavior)
	if err == nil && req.Kind == "speculative" && req.Behavior.FailSpeculative {
		category := req.Behavior.FailureCategory
		if category == "" || !knownFailureCategory(category) {
			category = protocol.FailureCompute
		}
		message := req.Behavior.FailureMessage
		if message == "" {
			message = "synthetic speculative worker failure"
		}
		err = &FailureError{Category: category, Message: message}
	}
	if err != nil {
		stopHeartbeats()
		<-heartbeatDone
		var failure *FailureError
		if errors.As(err, &failure) {
			return fail(failure.Category, failure.Message)
		}
		return fail(protocol.FailureCompute, err.Error())
	}

	stageCtx, stageCancel := context.WithTimeout(ctx, 5*time.Second)
	defer stageCancel()
	staged, err := k.Client.StageArtifact(stageCtx, req.CallbackBaseURL, req.AttemptID, req.Epoch,
		result.ChecksumSHA256, []byte(result.Output))
	if err != nil {
		stopHeartbeats()
		<-heartbeatDone
		return fail(protocol.FailureCompute, fmt.Sprintf("stage artifact failed: %v", err))
	}

	completeCtx, completeCancel := context.WithTimeout(ctx, 5*time.Second)
	defer completeCancel()
	err = k.Client.Complete(completeCtx, req.CallbackBaseURL, req.AttemptID, participant.CompleteRequest{
		Epoch:          req.Epoch,
		StagingID:      staged.StagingID,
		ChecksumSHA256: staged.ChecksumSHA256,
		SizeBytes:      staged.SizeBytes,
	})
	stopHeartbeats()
	<-heartbeatDone
	if err != nil {
		var httpErr *participant.HTTPError
		if errors.As(err, &httpErr) && httpErr.StatusCode == 409 {
			return nil
		}
		return err
	}
	return nil
}

func effectiveDelay(req participant.ExecuteRequest) int {
	if req.Kind == "original" && req.Behavior.OriginalDelayMS > 0 {
		return req.Behavior.OriginalDelayMS
	}
	if req.Kind == "speculative" && req.Behavior.SpeculativeDelayMS > 0 {
		return req.Behavior.SpeculativeDelayMS
	}
	return req.Behavior.DelayMS
}

func (k *Kernel) heartbeatLoop(ctx context.Context, done chan<- struct{}, req participant.ExecuteRequest, interval time.Duration, leaseFor time.Duration) {
	defer close(done)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			heartbeatCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
			err := k.Client.Heartbeat(heartbeatCtx, req.CallbackBaseURL, req.AttemptID, req.Epoch)
			cancel()
			if err != nil && ctx.Err() == nil {
				return
			}
		}
	}
}

type FailureError struct {
	Category string
	Message  string
}

func (e *FailureError) Error() string {
	return e.Category + ": " + e.Message
}

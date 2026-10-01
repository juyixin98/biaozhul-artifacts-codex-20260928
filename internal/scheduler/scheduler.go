package scheduler

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"opp300-speculative-batch/internal/protocol"
	"opp300-speculative-batch/internal/store"
)

type Scheduler struct {
	Store      *store.Store
	WorkerURL  string
	BaseURL    string
	Epoch      string
	Poll       time.Duration
	LeasePoll  time.Duration
	HTTPClient *http.Client
	Logger     *slog.Logger
	Now        func() time.Time
}

func (s *Scheduler) Run(ctx context.Context) {
	if s.Poll <= 0 {
		s.Poll = 100 * time.Millisecond
	}
	if s.LeasePoll <= 0 {
		s.LeasePoll = s.Poll
	}
	if s.HTTPClient == nil {
		s.HTTPClient = &http.Client{Timeout: 2 * time.Second}
	}
	if s.Now == nil {
		s.Now = time.Now
	}
	recovered, err := s.Store.RecoverInterrupted(ctx, s.Now(), s.Epoch)
	if err != nil {
		s.Logger.Error("startup recovery failed", "error", err, "epoch", s.Epoch)
	} else if recovered > 0 {
		s.Logger.Warn("interrupted attempts fenced", "count", recovered, "epoch", s.Epoch)
	}

	planTicker := time.NewTicker(s.Poll)
	leaseTicker := time.NewTicker(s.LeasePoll)
	defer planTicker.Stop()
	defer leaseTicker.Stop()

	for {
		if err := s.tick(ctx); err != nil && ctx.Err() == nil {
			s.Logger.Error("scheduler tick failed", "error", err)
		}
		select {
		case <-ctx.Done():
			return
		case <-leaseTicker.C:
			if err := s.Store.ReconcileExpired(ctx, s.Now(), s.Epoch); err != nil && ctx.Err() == nil {
				s.Logger.Error("lease reconciliation failed", "error", err)
			}
		case <-planTicker.C:
		}
	}
}

func (s *Scheduler) tick(ctx context.Context) error {
	plan, err := s.Store.PlanDispatch(ctx, s.Now(), s.Epoch)
	if err != nil {
		return err
	}
	if len(plan.Started) == 0 {
		return nil
	}
	return s.dispatch(ctx, plan.Started)
}

func (s *Scheduler) dispatch(ctx context.Context, attempts []protocol.Attempt) error {
	var wg sync.WaitGroup
	errCh := make(chan error, len(attempts))
	for _, attempt := range attempts {
		wg.Add(1)
		go func(attempt protocol.Attempt) {
			defer wg.Done()
			if err := s.dispatchOne(ctx, attempt); err != nil {
				errCh <- err
			}
		}(attempt)
	}
	wg.Wait()
	close(errCh)
	var first error
	for err := range errCh {
		if first == nil {
			first = err
		}
	}
	return first
}

func (s *Scheduler) dispatchOne(ctx context.Context, attempt protocol.Attempt) error {
	shard, err := s.Store.GetShard(ctx, attempt.ShardID)
	if err != nil {
		return err
	}
	policy, err := s.Store.GetPolicy(ctx, attempt.JobID)
	if err != nil {
		return err
	}
	payload := map[string]any{
		"attempt_id":            attempt.ID,
		"job_id":                attempt.JobID,
		"shard_id":              attempt.ShardID,
		"ordinal":               shard.Ordinal,
		"key":                   shard.Key,
		"attempt_number":        attempt.AttemptNumber,
		"kind":                  attempt.Kind,
		"epoch":                 s.Epoch,
		"input":                 shard.Input,
		"behavior":              shard.Behavior,
		"callback_base_url":     s.BaseURL,
		"lease_ms":              policy.LeaseMS,
		"heartbeat_interval_ms": policy.HeartbeatIntervalMS,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, s.WorkerURL+"/execute", bytes.NewReader(body))
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := s.HTTPClient.Do(request)
	if err != nil {
		return s.markDispatchFailure(ctx, attempt, err)
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return s.markDispatchFailure(ctx, attempt, fmt.Errorf("worker returned HTTP %d", response.StatusCode))
	}
	s.Logger.Info("attempt dispatched", "attempt_id", attempt.ID, "job_id", attempt.JobID,
		"shard_key", shard.Key, "kind", attempt.Kind, "attempt_number", attempt.AttemptNumber,
		"epoch", s.Epoch)
	return nil
}

func (s *Scheduler) markDispatchFailure(ctx context.Context, attempt protocol.Attempt, cause error) error {
	s.Logger.Error("dispatch failed", "attempt_id", attempt.ID, "error", cause)
	_, failErr := s.Store.FailAttempt(ctx, attempt.ID, store.AttemptFailure{
		Now:       s.Now(),
		Epoch:     s.Epoch,
		Category:  protocol.FailureWorker,
		Message:   cause.Error(),
		WorkerURL: s.WorkerURL,
	})
	if failErr != nil {
		return failErr
	}
	return nil
}

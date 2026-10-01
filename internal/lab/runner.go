package lab

import (
	"context"
	"errors"
	"fmt"

	"checkpointlab/internal/kernel"
	"checkpointlab/internal/oracle"
	"checkpointlab/internal/protocol"
	"checkpointlab/internal/replay"
	"checkpointlab/internal/source"
	"checkpointlab/internal/storage"
)

type StepObservation struct {
	Index         int                         `json:"index"`
	Name          string                      `json:"name"`
	Accepted      bool                        `json:"accepted"`
	ErrorCategory protocol.ErrorCategory      `json:"error_category,omitempty"`
	Message       string                      `json:"message,omitempty"`
	Result        *kernel.Result              `json:"result,omitempty"`
	Gate          map[string]kernel.GateStats `json:"gate,omitempty"`
}

type Report struct {
	JobID            string            `json:"job_id"`
	Scenario         source.Scenario   `json:"scenario"`
	Reference        oracle.Expected   `json:"reference"`
	Observations     []StepObservation `json:"observations"`
	CommittedState   storage.State     `json:"committed_state"`
	LiveState        storage.State     `json:"live_state"`
	CommittedOutputs []storage.Output  `json:"committed_outputs"`
	AllOutputs       []storage.Output  `json:"all_outputs"`
	Checkpoint       *storage.Snapshot `json:"checkpoint,omitempty"`
	Metrics          kernel.Metrics    `json:"metrics"`
	Failures         []Failure         `json:"failures"`
	Uncertainties    []string          `json:"uncertainties"`
}

type Failure struct {
	Category string `json:"category"`
	Step     string `json:"step"`
	Message  string `json:"message"`
}

type Runner struct {
	store storage.Store
}

func NewRunner(store storage.Store) *Runner {
	return &Runner{store: store}
}

func (r *Runner) Run(ctx context.Context, jobID string, sc source.Scenario, requestID string) (*kernel.Operator, Report, error) {
	if err := r.store.CreateJob(ctx, jobID, sc.Mode, nil); err != nil {
		return nil, Report{}, err
	}
	op, err := kernel.NewOperator(kernel.Config{
		JobID:         jobID,
		Mode:          sc.Mode,
		GateCapacity:  sc.GateCapacity,
		InflightLimit: sc.InflightLimit,
	}, r.store)
	if err != nil {
		return nil, Report{}, err
	}
	report, err := r.runSteps(ctx, op, jobID, sc, requestID)
	return op, report, err
}

func (r *Runner) Resume(ctx context.Context, jobID string, sc source.Scenario, requestID string) (*kernel.Operator, Report, error) {
	mode, err := r.store.JobMode(ctx, jobID)
	if err != nil {
		return nil, Report{}, err
	}
	if mode != sc.Mode {
		return nil, Report{}, fmt.Errorf("scenario mode %s does not match stored job mode %s", sc.Mode, mode)
	}
	op, err := kernel.NewOperator(kernel.Config{
		JobID:         jobID,
		Mode:          sc.Mode,
		GateCapacity:  sc.GateCapacity,
		InflightLimit: sc.InflightLimit,
	}, r.store)
	if err != nil {
		return nil, Report{}, err
	}
	point, err := op.Restore(ctx)
	if err != nil {
		return nil, Report{}, err
	}
	plan := replay.NewPlanner(sc).Build(point)
	_ = r.store.AppendEvent(ctx, storage.EventLogEntry{
		JobID: jobID, RequestID: requestID, Epoch: point.CommittedEpoch,
		Step: "replay_plan_built", Location: "replay",
		Detail: plan.Reason,
	})
	report := Report{JobID: jobID, Scenario: sc, Reference: oracle.ExpectedBoundary(sc)}
	for i, step := range plan.Steps {
		res, err := op.Process(ctx, step.Event(), requestID)
		obs := StepObservation{
			Index: i,
			Name:  step.Name,
			Gate: map[string]kernel.GateStats{
				"A": op.Gate("A").Stats(),
				"B": op.Gate("B").Stats(),
			},
		}
		if res.Accepted {
			cp := res
			obs.Accepted = true
			obs.Result = &cp
		}
		if err != nil {
			obs.Message = err.Error()
			var typed *protocol.TypedError
			if errors.As(err, &typed) {
				obs.ErrorCategory = typed.Category
			}
		}
		report.Observations = append(report.Observations, obs)
	}
	report.Metrics = op.Metrics()
	r.fillOutputsAndCheck(ctx, op, &report)
	r.Verify(ctx, &report)
	return op, report, nil
}

func (r *Runner) runSteps(ctx context.Context, op *kernel.Operator, jobID string, sc source.Scenario, requestID string) (Report, error) {
	report := Report{JobID: jobID, Scenario: sc, Reference: oracle.ExpectedBoundary(sc)}
	for i, step := range sc.Steps {
		var res kernel.Result
		var err error
		if step.Kind == protocol.KindRecord && step.ExpectError == protocol.CategoryAlignmentBlocked {
			err = op.Gate(step.Stream).TrySend(step.Event())
		} else {
			res, err = op.Process(ctx, step.Event(), requestID)
		}
		obs := StepObservation{
			Index: i,
			Name:  step.Name,
			Gate: map[string]kernel.GateStats{
				"A": op.Gate("A").Stats(),
				"B": op.Gate("B").Stats(),
			},
		}
		if res.Accepted {
			cp := res
			obs.Accepted = true
			obs.Result = &cp
		}
		if err != nil {
			obs.Message = err.Error()
			var typed *protocol.TypedError
			if errors.As(err, &typed) {
				obs.ErrorCategory = typed.Category
			}
		}
		if step.ExpectError != "" {
			if obs.ErrorCategory != step.ExpectError {
				report.Failures = append(report.Failures, Failure{
					Category: string(step.ExpectError),
					Step:     step.Name,
					Message:  fmt.Sprintf("expected %s, got %q", step.ExpectError, obs.Message),
				})
			}
		} else if err != nil {
			report.Failures = append(report.Failures, Failure{
				Category: "UNEXPECTED_STEP_ERROR", Step: step.Name, Message: err.Error(),
			})
		}
		report.Observations = append(report.Observations, obs)
	}
	report.Metrics = op.Metrics()
	r.fillOutputsAndCheck(ctx, op, &report)
	r.Verify(ctx, &report)
	return report, nil
}

func (r *Runner) fillOutputsAndCheck(ctx context.Context, op *kernel.Operator, report *Report) {
	report.LiveState = op.SnapshotState()
	committed, err := r.store.ListOutputs(ctx, report.JobID, true)
	if err == nil {
		report.CommittedOutputs = committed
		report.CommittedState = stateFromOutputs(committed)
	} else {
		report.Uncertainties = append(report.Uncertainties, "could not read committed outputs: "+err.Error())
	}
	all, err := r.store.ListOutputs(ctx, report.JobID, false)
	if err == nil {
		report.AllOutputs = all
	} else {
		report.Uncertainties = append(report.Uncertainties, "could not read outputs: "+err.Error())
	}
	if report.Metrics.CommittedEpoch > 0 {
		snap, ok, err := r.store.GetCheckpoint(ctx, report.JobID, report.Metrics.CommittedEpoch)
		if err != nil {
			report.Uncertainties = append(report.Uncertainties, "could not read checkpoint: "+err.Error())
		} else if ok {
			report.Checkpoint = &snap
		}
	}
}

func (r *Runner) Verify(ctx context.Context, report *Report) {
	if report.Metrics.CommittedEpoch != report.Reference.CommittedEpoch {
		report.Failures = append(report.Failures, Failure{
			Category: "COMMITTED_EPOCH_MISMATCH",
			Message:  fmt.Sprintf("got %d want %d", report.Metrics.CommittedEpoch, report.Reference.CommittedEpoch),
		})
	}
	if report.Metrics.PeakInflight != report.Reference.InflightPeak {
		report.Failures = append(report.Failures, Failure{
			Category: "INFLIGHT_OCCUPANCY_MISMATCH",
			Message:  fmt.Sprintf("got peak %d want %d", report.Metrics.PeakInflight, report.Reference.InflightPeak),
		})
	}
	for key, want := range report.Reference.CommittedState {
		if got := report.CommittedState[key]; got != want {
			report.Failures = append(report.Failures, Failure{
				Category: "REFERENCE_COUNT_MISMATCH",
				Step:     key,
				Message:  fmt.Sprintf("got %d want %d", got, want),
			})
		}
	}
	if got := report.Metrics.Gates["A"].BlockedAttempts; got != int64(report.Scenario.ExpectedReport.BlockedA) {
		report.Failures = append(report.Failures, Failure{
			Category: "BLOCKED_A_OCCUPANCY_MISMATCH",
			Message:  fmt.Sprintf("got %d want %d", got, report.Scenario.ExpectedReport.BlockedA),
		})
	}
	if got := report.Metrics.Gates["B"].BlockedAttempts; got != int64(report.Scenario.ExpectedReport.BlockedB) {
		report.Failures = append(report.Failures, Failure{
			Category: "BLOCKED_B_OCCUPANCY_MISMATCH",
			Message:  fmt.Sprintf("got %d want %d", got, report.Scenario.ExpectedReport.BlockedB),
		})
	}
}

func stateFromOutputs(outs []storage.Output) storage.State {
	state := storage.State{}
	for _, out := range outs {
		if out.Committed {
			state[out.Key] += out.Value
		}
	}
	return state
}

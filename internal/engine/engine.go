package engine

import (
	"context"
	"sync"

	"opp299/internal/kernel"
	"opp299/internal/protocol"
	"opp299/internal/replay"
	"opp299/internal/source"
)

type Result struct {
	Outputs []protocol.Output       `json:"outputs"`
	Metrics protocol.Metrics        `json:"metrics"`
	State   *protocol.State         `json:"state"`
	Failure *kernel.ClassifiedError `json:"failure,omitempty"`
}

func RunWorkload(ctx context.Context, cfg protocol.Config, store kernel.Store, workload source.Workload) (*kernel.Runner, Result) {
	runner, _ := kernel.NewRunner(cfg, store)
	if runner == nil {
		return nil, Result{
			State:   protocol.NewState(),
			Metrics: protocol.Metrics{Processed: map[string]int64{}, MaxBuffer: cfg.BackpressureLimit},
			Failure: kernel.Classify(cfg.Validate()),
		}
	}
	runCtx, cancel := context.WithCancel(ctx)
	var wg sync.WaitGroup
	for _, spec := range workload.Streams {
		spec := spec
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := source.Run(runCtx, spec, runner, 0, false); err == nil {
				runner.CloseStream(spec.ID)
			}
		}()
	}
	err := runner.Run(runCtx)
	if err != nil {
		runner.AbortSources()
	}
	cancel()
	wg.Wait()
	result := Result{
		Outputs: runner.Outputs(),
		Metrics: runner.SnapshotMetrics(),
		State:   runner.StateSnapshot(),
	}
	if err != nil {
		result.Failure = kernel.Classify(err)
	}
	return runner, result
}

func RecoverAndRun(ctx context.Context, cfg protocol.Config, store kernel.Store, cp protocol.Checkpoint, workload source.Workload) (*kernel.Runner, replay.Report, Result) {
	runner, replayReport, err := replay.Inflight(ctx, cfg, store, cp)
	if err != nil {
		result := Result{
			State:   runner.StateSnapshot(),
			Metrics: runner.SnapshotMetrics(),
			Failure: kernel.Classify(err),
		}
		return runner, replayReport, result
	}
	recoveredCfg := cfg
	recoveredCfg.FailCheckpointID = 0
	// A recovered run uses the restored state and waits for strictly newer
	// barriers. Existing producers skip through the checkpoint barrier.
	runCtx, cancel := context.WithCancel(ctx)
	var wg sync.WaitGroup
	for _, spec := range workload.Streams {
		spec := spec
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := source.Run(runCtx, spec, runner, cp.ID, true); err == nil {
				runner.CloseStream(spec.ID)
			}
		}()
	}
	runErr := runner.Run(runCtx)
	if runErr != nil {
		runner.AbortSources()
	}
	cancel()
	wg.Wait()
	result := Result{
		Outputs: runner.Outputs(),
		Metrics: runner.SnapshotMetrics(),
		State:   runner.StateSnapshot(),
	}
	if runErr != nil {
		result.Failure = kernel.Classify(runErr)
	}
	return runner, replayReport, result
}

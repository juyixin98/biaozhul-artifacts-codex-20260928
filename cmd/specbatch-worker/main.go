package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"

	"specbatch/internal/config"
	"specbatch/internal/logging"
	"specbatch/internal/worker"
)

func main() {
	cfg, err := config.WorkerFromFlags(os.Args[1:])
	if err != nil {
		os.Exit(2)
	}
	logger, err := logging.New(cfg.LogPath, true)
	if err != nil {
		panic(err)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	w := &worker.Worker{
		ID: cfg.WorkerID, Client: worker.NewClient(cfg.CoordinatorURL),
		StagingRoot: cfg.StagingRoot, Log: logger, PollEvery: cfg.PollInterval,
	}
	logger.Info("worker_ready", map[string]any{
		"worker_id": cfg.WorkerID, "coordinator_url": cfg.CoordinatorURL,
		"staging_root": cfg.StagingRoot,
	})
	if err := w.Run(ctx); err != nil {
		logger.Error("worker_stopped_with_error", map[string]any{"error": err.Error()})
		os.Exit(1)
	}
	logger.Info("worker_stopped", map[string]any{"worker_id": cfg.WorkerID})
}

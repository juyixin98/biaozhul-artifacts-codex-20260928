package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"opp300-speculative-batch/internal/artifacts"
	"opp300-speculative-batch/internal/config"
	"opp300-speculative-batch/internal/coordinator"
	"opp300-speculative-batch/internal/idgen"
	"opp300-speculative-batch/internal/scheduler"
	"opp300-speculative-batch/internal/store"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	cfg, err := config.Load()
	if err != nil {
		logger.Error("configuration error", "error", err)
		os.Exit(2)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	poolStore, closePool, err := store.New(ctx, cfg.DatabaseURL)
	if err != nil {
		logger.Error("database connection failed", "error", err)
		os.Exit(1)
	}
	defer closePool()
	if err := poolStore.Ping(ctx); err != nil {
		logger.Error("database ping failed", "error", err)
		os.Exit(1)
	}
	if err := poolStore.Migrate(ctx, migrationPath("migrations")); err != nil {
		logger.Error("database migration failed", "error", err)
		os.Exit(1)
	}
	artifactStore, err := artifacts.NewStore(cfg.ArtifactDir)
	if err != nil {
		logger.Error("artifact store initialization failed", "error", err)
		os.Exit(1)
	}
	if err := artifactStore.CleanupStaging(); err != nil {
		logger.Error("staging cleanup failed", "error", err)
		os.Exit(1)
	}

	epoch, err := idgen.New("epoch")
	if err != nil {
		logger.Error("epoch generation failed", "error", err)
		os.Exit(1)
	}
	logger.Info("coordinator starting",
		"version", "local-1.0.0", "epoch", epoch, "http_addr", cfg.HTTPAddr,
		"database", redactDatabaseURL(cfg.DatabaseURL), "worker_url", cfg.WorkerURL,
		"artifact_dir", cfg.ArtifactDir, "policy", cfg.DefaultPolicy)

	server := coordinator.NewServer(poolStore, artifactStore, logger, epoch, cfg.BaseURL)
	httpServer := &http.Server{
		Addr:         cfg.HTTPAddr,
		Handler:      server.Routes(),
		ReadTimeout:  cfg.HTTPReadTimeout,
		WriteTimeout: cfg.HTTPWriteTimeout,
	}

	schedulerCtx, cancelScheduler := context.WithCancel(context.Background())
	schedulerDone := make(chan struct{})
	go func() {
		defer close(schedulerDone)
		(&scheduler.Scheduler{
			Store:     poolStore,
			WorkerURL: cfg.WorkerURL,
			BaseURL:   cfg.BaseURL,
			Epoch:     epoch,
			Poll:      cfg.PollInterval,
			LeasePoll: cfg.LeaseInterval,
			Logger:    logger.With("component", "scheduler"),
		}).Run(schedulerCtx)
	}()

	serverErrors := make(chan error, 1)
	go func() {
		if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			serverErrors <- err
		}
	}()

	select {
	case <-ctx.Done():
		logger.Info("shutdown signal received", "epoch", epoch)
	case err := <-serverErrors:
		logger.Error("http server failed", "error", err)
		cancelScheduler()
		os.Exit(1)
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownTimeout)
	defer cancel()
	if err := httpServer.Shutdown(shutdownCtx); err != nil {
		logger.Error("http shutdown failed", "error", err)
	}
	cancelScheduler()
	<-schedulerDone
	logger.Info("coordinator stopped", "epoch", epoch)
}

func redactDatabaseURL(value string) string {
	if len(value) <= 12 {
		return value
	}
	for _, scheme := range []string{"postgres://", "postgresql://"} {
		if len(value) > len(scheme)+3 && value[:len(scheme)] == scheme {
			return fmt.Sprintf("%s***@%s", scheme, hostFromURL(value[len(scheme):]))
		}
	}
	return value
}

func hostFromURL(afterScheme string) string {
	for i := 0; i < len(afterScheme); i++ {
		if afterScheme[i] == '/' {
			return afterScheme[i+1:]
		}
	}
	return "database"
}

func migrationPath(configured string) string {
	if _, err := os.Stat(configured); err == nil {
		return configured
	}
	executable, err := os.Executable()
	if err != nil {
		return configured
	}
	candidate := filepath.Join(filepath.Dir(executable), "..", "migrations")
	if _, err := os.Stat(candidate); err == nil {
		return candidate
	}
	return configured
}

package main

import (
	"context"
	"errors"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"specbatch/internal/config"
	"specbatch/internal/coordinator"
	"specbatch/internal/logging"
	"specbatch/internal/model"
	"specbatch/internal/store"
)

func main() {
	cfg, err := config.CoordinatorFromFlags(os.Args[1:])
	if err != nil {
		os.Exit(2)
	}
	logger, err := logging.New(cfg.LogPath, true)
	if err != nil {
		panic(err)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	st, err := store.Open(ctx, cfg.DatabaseURL)
	if err != nil {
		logger.Error("database_unavailable", map[string]any{"error": err.Error(), "dsn": config.RedactedDSN(cfg.DatabaseURL)})
		os.Exit(1)
	}
	defer st.Close()
	epoch, err := st.Bootstrap(ctx)
	if err != nil {
		logger.Error("bootstrap_failed", map[string]any{"error": err.Error()})
		os.Exit(1)
	}
	if err := os.MkdirAll(cfg.ArtifactDir, 0o755); err != nil {
		logger.Error("artifact_dir_failed", map[string]any{"error": err.Error()})
		os.Exit(1)
	}
	logger.Info("coordinator_ready", map[string]any{
		"addr": cfg.HTTPAddr, "run_epoch": epoch, "protocol": cfg.ProtocolVersion,
		"artifact_dir": cfg.ArtifactDir, "dsn": config.RedactedDSN(cfg.DatabaseURL),
		"kernel_version": "v1.0.0", "failure_category": model.FailureCoordinatorRestart,
	})

	srv := coordinator.NewServer(st, cfg.ArtifactDir, epoch, logger)
	runner := coordinator.NewRunner(st, epoch, cfg.ScheduleTick, cfg.ArtifactDir, logger)
	go runner.Run(ctx)

	httpServer := &http.Server{Addr: cfg.HTTPAddr, Handler: srv.Routes(), ReadHeaderTimeout: 2 * time.Second}
	go func() {
		if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error("http_server_failed", map[string]any{"error": err.Error()})
			stop()
		}
	}()
	<-ctx.Done()
	shutdown, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_ = httpServer.Shutdown(shutdown)
	logger.Info("coordinator_stopped", map[string]any{"run_epoch": epoch})
}

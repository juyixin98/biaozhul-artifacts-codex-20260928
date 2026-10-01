package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"opp300-speculative-batch/internal/participant"
	"opp300-speculative-batch/internal/worker"
)

func main() {
	addr := env("WORKER_HTTP_ADDR", "127.0.0.1:18081")
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	kernel := &worker.Kernel{Client: participant.NewClient()}
	workerServer := worker.NewServer(kernel, logger.With("component", "worker"))
	httpServer := &http.Server{
		Addr:              addr,
		Handler:           workerServer.Routes(),
		ReadHeaderTimeout: 5 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	serverErrors := make(chan error, 1)
	go func() {
		logger.Info("synthetic worker starting", "version", "local-1.0.0", "http_addr", addr)
		if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			serverErrors <- err
		}
	}()

	select {
	case <-ctx.Done():
	case err := <-serverErrors:
		logger.Error("worker http server failed", "error", err)
		os.Exit(1)
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := httpServer.Shutdown(shutdownCtx); err != nil {
		logger.Error("worker shutdown failed", "error", err)
	}
	workerServer.Shutdown()
	logger.Info("synthetic worker stopped")
}

func env(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

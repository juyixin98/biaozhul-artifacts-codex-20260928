package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"diststats/internal/httpapi"
	"diststats/internal/postgres"
	"diststats/internal/service"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	slog.SetDefault(logger)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	databaseURL := getenv("DATABASE_URL", "postgres://admin@localhost:5432/opp298?sslmode=disable")
	addr := getenv("HTTP_ADDR", ":8080")
	migrationsDir := getenv("MIGRATIONS_DIR", "migrations")
	maxConns, _ := strconv.ParseInt(getenv("PG_MAX_CONNS", "8"), 10, 32)

	pool, err := postgres.Connect(ctx, databaseURL, int32(maxConns))
	if err != nil {
		logger.Error("database connect failed", "error", err)
		os.Exit(1)
	}
	defer pool.Close()

	pingContext, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if err := pool.Ping(pingContext); err != nil {
		logger.Error("database ping failed", "error", err)
		os.Exit(1)
	}
	if err := postgres.Migrate(ctx, pool, migrationsDir); err != nil {
		logger.Error("database migration failed", "error", err)
		os.Exit(1)
	}

	store := postgres.NewStore(pool)
	svc := service.New(store)
	server := &http.Server{
		Addr:              addr,
		Handler:           httpapi.New(svc, logger).Handler(),
		ReadHeaderTimeout: 5 * time.Second,
	}

	go func() {
		logger.Info("server listening", "addr", addr)
		if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error("server failed", "error", err)
			stop()
		}
	}()

	<-ctx.Done()
	shutdownContext, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer shutdownCancel()
	if err := server.Shutdown(shutdownContext); err != nil {
		logger.Error("server shutdown failed", "error", err)
		os.Exit(1)
	}
	logger.Info("server stopped")
}

func getenv(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

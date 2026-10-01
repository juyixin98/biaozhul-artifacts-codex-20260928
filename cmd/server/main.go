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

	"diststats/internal/config"
	"diststats/internal/httpapi"
	"diststats/internal/service"
	"diststats/internal/store"
)

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{}))
	cfg := config.Load()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	db, err := store.Open(ctx, cfg.DatabaseURL)
	if err != nil {
		log.Error("database connect failed", "error", err)
		os.Exit(1)
	}
	defer db.Close()
	if err := db.Migrate(ctx); err != nil {
		log.Error("database migration failed", "error", err)
		os.Exit(1)
	}

	svc := service.New(db, service.Limits{
		MaxDimensions:   cfg.MaxDimensions,
		MaxObservations: cfg.MaxObservations,
	})
	server := &http.Server{
		Addr:              cfg.Addr,
		Handler:           httpapi.New(svc, log),
		ReadHeaderTimeout: 5 * time.Second,
	}

	go func() {
		log.Info("server listening", "addr", cfg.Addr)
		if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Error("server failed", "error", err)
			os.Exit(1)
		}
	}()

	<-ctx.Done()
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := server.Shutdown(shutdownCtx); err != nil {
		log.Error("graceful shutdown failed", "error", err)
		os.Exit(1)
	}
	log.Info("server stopped")
}

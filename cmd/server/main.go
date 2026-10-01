package main

import (
	"context"
	"errors"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"opp299/internal/config"
	"opp299/internal/httpapi"
	"opp299/internal/storage"
)

func main() {
	configPath := flag.String("config", "configs/server.json", "path to JSON configuration")
	flag.Parse()

	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Fatalf("load config: %v", err)
	}
	var store storage.CheckpointStore
	if cfg.Storage == "postgres" {
		pgStore, err := storage.NewPostgresStore(cfg.DatabaseURL, cfg.MigrationPath)
		if err != nil {
			log.Fatalf("connect storage: %v", err)
		}
		defer pgStore.Close()
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		if err := pgStore.Ping(ctx); err != nil {
			cancel()
			log.Fatalf("ping postgres: %v", err)
		}
		if err := pgStore.Init(ctx); err != nil {
			cancel()
			log.Fatalf("initialize schema: %v", err)
		}
		cancel()
		store = pgStore
	} else {
		store = storage.NewMemoryStore()
	}

	server := &http.Server{
		Addr:              cfg.HTTPAddr,
		Handler:           httpapi.New(cfg, store).Routes(),
		ReadHeaderTimeout: 5 * time.Second,
	}
	go func() {
		log.Printf("listening on %s storage=%s request logs include request_id", cfg.HTTPAddr, cfg.Storage)
		if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("http server: %v", err)
		}
	}()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	<-ctx.Done()
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := server.Shutdown(shutdownCtx); err != nil {
		log.Fatalf("shutdown: %v", err)
	}
}

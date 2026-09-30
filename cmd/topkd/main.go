package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"opp297/topk/internal/config"
	"opp297/topk/internal/fixture"
	"opp297/topk/internal/participant"
	"opp297/topk/internal/service"
	"opp297/topk/internal/storage"
)

func main() {
	logger := log.New(os.Stdout, "topkd ", log.LstdFlags|log.Lmicroseconds)
	if len(os.Args) < 3 || (os.Args[1] != "run" && os.Args[1] != "init-db" && os.Args[1] != "seed") {
		fatalf(logger, "usage: topkd {run|init-db|seed} config.json")
	}

	cfg, err := config.Load(os.Args[2])
	if err != nil {
		fatalf(logger, "config: %v", err)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	switch os.Args[1] {
	case "init-db":
		runInitDB(ctx, logger, cfg)
	case "seed":
		runSeed(ctx, logger, cfg)
	case "run":
		run(ctx, logger, cfg)
	}
}

func runInitDB(ctx context.Context, logger *log.Logger, cfg config.Config) {
	store, closeStore, err := storage.Open(ctx, cfg.DatabaseURL)
	if err != nil {
		fatalf(logger, "open database: %v", err)
	}
	defer closeStore()
	if err := store.EnsureSchema(ctx); err != nil {
		fatalf(logger, "schema: %v", err)
	}
	logger.Printf("schema ready")
}

func runSeed(ctx context.Context, logger *log.Logger, cfg config.Config) {
	store, closeStore, err := storage.Open(ctx, cfg.DatabaseURL)
	if err != nil {
		fatalf(logger, "open database: %v", err)
	}
	defer closeStore()
	records := fixture.Default()
	if cfg.Generation != "" && cfg.Generation != string(fixture.DefaultGeneration) {
		fatalf(logger, "this local fixture only supports generation %s", fixture.DefaultGeneration)
	}
	if err := store.SeedGeneration(ctx, fixture.DefaultGeneration, records); err != nil {
		fatalf(logger, "seed: %v", err)
	}
	logger.Printf("seed ready generation=%s items=%d", fixture.DefaultGeneration, len(records))
}

func run(ctx context.Context, logger *log.Logger, cfg config.Config) {
	store, closeStore, err := storage.Open(ctx, cfg.DatabaseURL)
	if err != nil {
		fatalf(logger, "open database: %v", err)
	}
	defer closeStore()

	if cfg.Role == "shard" {
		runServer(ctx, logger, cfg.ListenAddress, (&participant.Server{
			Store: store, ShardID: cfg.ShardID, BoundSecret: []byte(cfg.BoundSecret),
			Fault: cfg.Fault, AllowFault: cfg.AllowTestFaults, Logger: logger,
		}).Routes())
		return
	}

	if cfg.Role != "coordinator" && cfg.Role != "combined" {
		fatalf(logger, "role must be coordinator, shard, or combined")
	}

	servers := map[int]*http.Server{}
	urls := make(map[int]string, len(cfg.Shards))
	if cfg.Role == "combined" {
		for _, shard := range cfg.Shards {
			if shard.URL == "" || shard.ListenAddress == "" {
				fatalf(logger, "combined shard %d requires url and listen_address", shard.ID)
			}
			urls[shard.ID] = shard.URL
		}
	} else {
		for _, shard := range cfg.Shards {
			if shard.URL == "" {
				fatalf(logger, "shard %d requires url", shard.ID)
			}
			urls[shard.ID] = shard.URL
		}
	}

	ids := make([]int, 0, len(cfg.Shards))
	for _, shard := range cfg.Shards {
		ids = append(ids, shard.ID)
		if cfg.Role == "combined" {
			srv := &http.Server{
				Addr:              shard.ListenAddress,
				Handler:           (&participant.Server{Store: store, ShardID: shard.ID, BoundSecret: []byte(cfg.BoundSecret), Fault: shard.Fault, AllowFault: cfg.AllowTestFaults, Logger: logger}).Routes(),
				ReadHeaderTimeout: 2 * time.Second,
			}
			servers[shard.ID] = srv
			go func(s *http.Server) {
				logger.Printf("shard listening on %s", s.Addr)
				if err := s.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
					fatalf(logger, "shard http: %v", err)
				}
			}(srv)
		}
	}

	coordinator := &service.Coordinator{
		Store: store, ShardIDs: ids, ShardURLs: urls,
		PageSize: cfg.PageSize, Timeout: cfg.ShardTimeout.Duration(),
		BoundSecret: []byte(cfg.BoundSecret), Logger: logger,
	}
	coordinatorServer := &http.Server{
		Addr:              cfg.ListenAddress,
		Handler:           coordinator.Routes(),
		ReadHeaderTimeout: 2 * time.Second,
	}
	go func() {
		logger.Printf("coordinator listening on %s", coordinatorServer.Addr)
		if err := coordinatorServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			fatalf(logger, "coordinator http: %v", err)
		}
	}()

	<-ctx.Done()
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := coordinatorServer.Shutdown(shutdownCtx); err != nil {
		logger.Printf("coordinator shutdown error=%v", err)
	}
	for _, srv := range servers {
		if err := srv.Shutdown(shutdownCtx); err != nil {
			logger.Printf("shard shutdown error=%v", err)
		}
	}
}

func runServer(ctx context.Context, logger *log.Logger, addr string, handler http.Handler) {
	srv := &http.Server{Addr: addr, Handler: handler, ReadHeaderTimeout: 2 * time.Second}
	go func() {
		logger.Printf("listening on %s", addr)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			fatalf(logger, "http: %v", err)
		}
	}()
	<-ctx.Done()
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		logger.Printf("shutdown error=%v", err)
	}
}

func fatalf(logger *log.Logger, format string, args ...any) {
	logger.Output(2, fmt.Sprintf(format, args...))
	os.Exit(1)
}

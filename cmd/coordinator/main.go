package main

import (
	"context"
	"flag"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"topk-demo/internal/config"
	"topk-demo/internal/coordinator"
	"topk-demo/internal/safolog"
	"topk-demo/internal/state"
)

func main() {
	configPath := flag.String("config", "configs/local.json", "configuration file")
	flag.Parse()

	log := safolog.New("coordinator")
	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Error("load config", "error", err)
		os.Exit(1)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	store, err := state.NewPostgresStore(ctx, cfg.DatabaseDSN)
	if err != nil {
		log.Error("connect state database", "error", err)
		os.Exit(1)
	}
	defer store.Close()

	coordCfg := coordinator.Config{
		Generation:   cfg.Coordinator.Generation,
		DefaultK:     cfg.Coordinator.DefaultK,
		DefaultPage:  cfg.Coordinator.DefaultPageSize,
		ShardTimeout: cfg.ShardTimeout(),
		Shards:       make([]coordinator.ShardConfig, 0, len(cfg.Coordinator.Shards)),
	}
	for _, shard := range cfg.Coordinator.Shards {
		coordCfg.Shards = append(coordCfg.Shards, coordinator.ShardConfig{ID: shard.ID, URL: shard.URL})
	}
	server := &http.Server{
		Addr:              cfg.Coordinator.Listen,
		Handler:           coordinator.New(coordCfg, store, log).Routes(),
		ReadHeaderTimeout: 5 * time.Second,
	}
	go func() {
		log.Info("coordinator listening", "addr", cfg.Coordinator.Listen)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Error("server failed", "error", err)
			os.Exit(1)
		}
	}()
	<-ctx.Done()
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = server.Shutdown(shutdownCtx)
}

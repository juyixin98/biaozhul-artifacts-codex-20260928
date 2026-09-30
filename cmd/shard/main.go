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
	"topk-demo/internal/safolog"
	"topk-demo/internal/shardserver"
	"topk-demo/internal/shardstore"
)

func main() {
	configPath := flag.String("config", "configs/local.json", "configuration file")
	shardID := flag.String("shard", "", "shard ID to serve")
	flag.Parse()

	log := safolog.New("shard")
	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Error("load config", "error", err)
		os.Exit(1)
	}
	listen := ""
	for _, shard := range cfg.Shards {
		if shard.ID == *shardID || shard.ShardID == *shardID {
			listen = shard.Listen
			*shardID = shard.ID
			break
		}
	}
	if listen == "" {
		log.Error("unknown shard", "shard_id", *shardID)
		os.Exit(1)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	store, err := shardstore.NewPostgresStore(ctx, cfg.DatabaseDSN)
	if err != nil {
		log.Error("connect database", "error", err)
		os.Exit(1)
	}
	defer store.Close()

	server := &http.Server{
		Addr:              listen,
		Handler:           shardserver.New(store, log).Routes(),
		ReadHeaderTimeout: 5 * time.Second,
	}
	go func() {
		log.Info("shard listening", "shard_id", *shardID, "addr", listen)
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

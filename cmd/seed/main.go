package main

import (
	"context"
	"flag"
	"os"

	"topk-demo/internal/config"
	"topk-demo/internal/safolog"
	"topk-demo/internal/seed"
)

func main() {
	configPath := flag.String("config", "configs/local.json", "configuration file")
	flag.Parse()
	log := safolog.New("seed")
	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Error("load config", "error", err)
		os.Exit(1)
	}
	if err := seed.Run(context.Background(), cfg.DatabaseDSN); err != nil {
		log.Error("seed failed", "error", err)
		os.Exit(1)
	}
	log.Info("seed complete", "generation", "gen-fixture-001")
}

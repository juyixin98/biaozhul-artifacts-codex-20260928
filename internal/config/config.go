package config

import (
	"encoding/json"
	"fmt"
	"os"
	"time"
)

type Shard struct {
	ID      string `json:"id"`
	Listen  string `json:"listen"`
	ShardID string `json:"shard_id"`
}

type Coordinator struct {
	Listen          string `json:"listen"`
	Generation      string `json:"generation"`
	DefaultK        int    `json:"default_k"`
	DefaultPageSize int    `json:"default_page_size"`
	ShardTimeoutMS  int    `json:"shard_timeout_ms"`
	Shards          []struct {
		ID  string `json:"id"`
		URL string `json:"url"`
	} `json:"shards"`
}

type App struct {
	DatabaseDSN string      `json:"database_dsn"`
	Shards      []Shard     `json:"shards"`
	Coordinator Coordinator `json:"coordinator"`
}

func Load(path string) (App, error) {
	var cfg App
	data, err := os.ReadFile(path)
	if err != nil {
		return cfg, fmt.Errorf("read config: %w", err)
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return cfg, fmt.Errorf("parse config: %w", err)
	}
	if cfg.DatabaseDSN == "" {
		return cfg, fmt.Errorf("database_dsn is required")
	}
	if len(cfg.Shards) == 0 || len(cfg.Coordinator.Shards) == 0 {
		return cfg, fmt.Errorf("at least one shard is required")
	}
	return cfg, nil
}

func (a App) ShardTimeout() time.Duration {
	if a.Coordinator.ShardTimeoutMS <= 0 {
		return 750 * time.Millisecond
	}
	return time.Duration(a.Coordinator.ShardTimeoutMS) * time.Millisecond
}

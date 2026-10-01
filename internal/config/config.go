package config

import (
	"encoding/json"
	"fmt"
	"os"
)

type Config struct {
	HTTPAddr                 string `json:"http_addr"`
	DatabaseURL              string `json:"database_url"`
	MigrationPath            string `json:"migration_path"`
	Storage                  string `json:"storage"`
	DefaultBackpressureLimit int    `json:"default_backpressure_limit"`
}

func Default() Config {
	return Config{
		HTTPAddr:                 ":8099",
		DatabaseURL:              "host=/var/run/postgresql dbname=opp299 user=admin sslmode=disable",
		MigrationPath:            "migrations/001_checkpoints.sql",
		Storage:                  "memory",
		DefaultBackpressureLimit: 4,
	}
}

func Load(path string) (Config, error) {
	cfg := Default()
	if path != "" {
		data, err := os.ReadFile(path)
		if err != nil {
			return cfg, err
		}
		if err := json.Unmarshal(data, &cfg); err != nil {
			return cfg, err
		}
	}
	if value := os.Getenv("HTTP_ADDR"); value != "" {
		cfg.HTTPAddr = value
	}
	if value := os.Getenv("DATABASE_URL"); value != "" {
		cfg.DatabaseURL = value
	}
	if value := os.Getenv("STORAGE"); value != "" {
		cfg.Storage = value
	}
	if value := os.Getenv("MIGRATION_PATH"); value != "" {
		cfg.MigrationPath = value
	}
	if cfg.Storage != "memory" && cfg.Storage != "postgres" {
		return cfg, fmt.Errorf("storage must be memory or postgres")
	}
	if cfg.DefaultBackpressureLimit <= 0 {
		return cfg, fmt.Errorf("default_backpressure_limit must be positive")
	}
	return cfg, nil
}

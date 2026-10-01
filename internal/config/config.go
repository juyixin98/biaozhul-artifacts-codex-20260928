package config

import (
	"fmt"
	"os"
	"strconv"
	"time"

	"opp300-speculative-batch/internal/protocol"
)

type Config struct {
	DatabaseURL      string
	HTTPAddr         string
	BaseURL          string
	ArtifactDir      string
	WorkerURL        string
	PollInterval     time.Duration
	LeaseInterval    time.Duration
	DefaultPolicy    protocol.Policy
	ShutdownTimeout  time.Duration
	HTTPReadTimeout  time.Duration
	HTTPWriteTimeout time.Duration
}

func Load() (Config, error) {
	poll := envDuration("COORDINATOR_POLL_INTERVAL", 100*time.Millisecond)
	lease := envDuration("COORDINATOR_LEASE_INTERVAL", 100*time.Millisecond)
	cfg := Config{
		DatabaseURL:      env("DATABASE_URL", "postgres:///opp300_dev?host=/var/run/postgresql&sslmode=disable"),
		HTTPAddr:         env("HTTP_ADDR", "127.0.0.1:18080"),
		BaseURL:          env("COORDINATOR_BASE_URL", "http://127.0.0.1:18080"),
		ArtifactDir:      env("ARTIFACT_DIR", "./var/artifacts"),
		WorkerURL:        env("WORKER_URL", "http://127.0.0.1:18081"),
		PollInterval:     poll,
		LeaseInterval:    lease,
		ShutdownTimeout:  envDuration("SHUTDOWN_TIMEOUT", 5*time.Second),
		HTTPReadTimeout:  envDuration("HTTP_READ_TIMEOUT", 10*time.Second),
		HTTPWriteTimeout: envDuration("HTTP_WRITE_TIMEOUT", 10*time.Second),
		DefaultPolicy: protocol.Policy{
			MaxAttemptsPerShard: envInt("MAX_ATTEMPTS_PER_SHARD", 2),
			MaxActiveAttempts:   envInt("MAX_ACTIVE_ATTEMPTS", 8),
			LeaseMS:             envInt("LEASE_MS", 5000),
			SpeculateAfterMS:    envInt("SPECULATE_AFTER_MS", 1500),
			HeartbeatIntervalMS: envInt("HEARTBEAT_INTERVAL_MS", 500),
		}.Normalized(),
	}
	if cfg.DatabaseURL == "" {
		return cfg, fmt.Errorf("DATABASE_URL is required")
	}
	if cfg.ArtifactDir == "" {
		return cfg, fmt.Errorf("ARTIFACT_DIR is required")
	}
	return cfg, nil
}

func env(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func envInt(key string, fallback int) int {
	raw := os.Getenv(key)
	if raw == "" {
		return fallback
	}
	value, err := strconv.Atoi(raw)
	if err != nil {
		return fallback
	}
	return value
}

func envDuration(key string, fallback time.Duration) time.Duration {
	raw := os.Getenv(key)
	if raw == "" {
		return fallback
	}
	value, err := time.ParseDuration(raw)
	if err != nil {
		return fallback
	}
	return value
}

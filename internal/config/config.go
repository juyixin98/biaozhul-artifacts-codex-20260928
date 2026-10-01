package config

import (
	"os"
	"strings"
)

type Config struct {
	Addr        string
	DatabaseURL string
	StoreKind   string
}

func Load() Config {
	cfg := Config{
		Addr:        env("ADDR", ":8080"),
		DatabaseURL: os.Getenv("DATABASE_URL"),
		StoreKind:   strings.ToLower(env("STORE", "memory")),
	}
	if cfg.DatabaseURL != "" {
		cfg.StoreKind = "postgres"
	}
	return cfg
}

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

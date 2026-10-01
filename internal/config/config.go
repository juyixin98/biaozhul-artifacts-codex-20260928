package config

import (
	"os"
	"strconv"
)

type Config struct {
	Addr            string
	DatabaseURL     string
	MaxDimensions   int
	MaxObservations int
}

func Load() Config {
	c := Config{
		Addr:            ":8080",
		DatabaseURL:     "postgres:///opp298?sslmode=disable",
		MaxDimensions:   64,
		MaxObservations: 100_000,
	}
	if v := os.Getenv("DATABASE_URL"); v != "" {
		c.DatabaseURL = v
	}
	if v := os.Getenv("HTTP_ADDR"); v != "" {
		c.Addr = v
	}
	if v := os.Getenv("MAX_DIMENSIONS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			c.MaxDimensions = n
		}
	}
	if v := os.Getenv("MAX_OBSERVATIONS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			c.MaxObservations = n
		}
	}
	return c
}

package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"time"
)

type Shard struct {
	ID            int    `json:"id"`
	URL           string `json:"url,omitempty"`
	ListenAddress string `json:"listen_address,omitempty"`
	Fault         string `json:"fault,omitempty"`
}

type Config struct {
	Role            string       `json:"role"`
	ListenAddress   string       `json:"listen_address"`
	DatabaseURL     string       `json:"database_url"`
	BoundSecret     string       `json:"bound_secret"`
	Generation      string       `json:"generation,omitempty"`
	ShardID         int          `json:"shard_id,omitempty"`
	PageSize        int          `json:"page_size"`
	ShardTimeout    JSONDuration `json:"shard_timeout"`
	Shards          []Shard      `json:"shards,omitempty"`
	Fault           string       `json:"fault,omitempty"`
	AllowTestFaults bool         `json:"allow_test_faults,omitempty"`
}

type JSONDuration time.Duration

func (d JSONDuration) Duration() time.Duration { return time.Duration(d) }

func (d *JSONDuration) UnmarshalJSON(data []byte) error {
	var text string
	if err := json.Unmarshal(data, &text); err != nil {
		return err
	}
	parsed, err := time.ParseDuration(text)
	if err != nil {
		return err
	}
	*d = JSONDuration(parsed)
	return nil
}

func Load(path string) (Config, error) {
	var cfg Config
	data, err := os.ReadFile(path)
	if err != nil {
		return cfg, err
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return cfg, fmt.Errorf("parse config: %w", err)
	}
	if cfg.ListenAddress == "" {
		cfg.ListenAddress = ":8080"
	}
	if cfg.PageSize == 0 {
		cfg.PageSize = 16
	}
	if cfg.ShardTimeout == 0 {
		cfg.ShardTimeout = JSONDuration(2 * time.Second)
	}
	if cfg.BoundSecret == "" {
		return cfg, errors.New("bound_secret is required")
	}
	if len(cfg.BoundSecret) < 16 {
		return cfg, errors.New("bound_secret must contain at least 16 bytes")
	}
	if cfg.DatabaseURL == "" {
		return cfg, errors.New("database_url is required")
	}
	return cfg, nil
}

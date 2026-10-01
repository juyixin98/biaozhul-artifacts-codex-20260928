package config

import (
	"errors"
	"flag"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"
)

type Coordinator struct {
	HTTPAddr        string
	DatabaseURL     string
	ArtifactDir     string
	ScheduleTick    time.Duration
	HeartbeatTTL    time.Duration
	LogPath         string
	ProtocolVersion string
}

type Worker struct {
	CoordinatorURL string
	DatabaseURL    string
	WorkerID       string
	StagingRoot    string
	PollInterval   time.Duration
	LogPath        string
}

func CoordinatorFromFlags(args []string) (Coordinator, error) {
	fs := flag.NewFlagSet("coordinator", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	cfg := Coordinator{}
	fs.StringVar(&cfg.HTTPAddr, "http-addr", env("COORDINATOR_HTTP_ADDR", "127.0.0.1:18080"), "HTTP listen address")
	fs.StringVar(&cfg.DatabaseURL, "database-url", env("DATABASE_URL", "postgres://postgres@127.0.0.1:5432/postgres?sslmode=disable"), "PostgreSQL URL")
	fs.StringVar(&cfg.ArtifactDir, "artifact-dir", env("ARTIFACT_DIR", "./run/artifacts"), "published artifact directory")
	fs.DurationVar(&cfg.ScheduleTick, "schedule-tick", durationEnv("SCHEDULE_TICK", 20*time.Millisecond), "scheduler interval")
	fs.DurationVar(&cfg.HeartbeatTTL, "heartbeat-ttl", durationEnv("HEARTBEAT_TTL", 120*time.Millisecond), "running attempt heartbeat TTL")
	fs.StringVar(&cfg.LogPath, "log-path", env("COORDINATOR_LOG_PATH", "./run/coordinator.log"), "structured log file")
	if err := fs.Parse(args); err != nil {
		return cfg, err
	}
	cfg.ProtocolVersion = "specbatch/v1"
	return cfg, validate(cfg.DatabaseURL == "", "database-url is required", cfg.ArtifactDir == "", "artifact-dir is required", cfg.ScheduleTick <= 0, "schedule-tick must be positive", cfg.HeartbeatTTL <= 0, "heartbeat-ttl must be positive")
}

func WorkerFromFlags(args []string) (Worker, error) {
	fs := flag.NewFlagSet("worker", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	cfg := Worker{}
	defaultID := env("WORKER_ID", "worker-"+strconv.Itoa(os.Getpid()))
	fs.StringVar(&cfg.CoordinatorURL, "coordinator-url", env("COORDINATOR_URL", "http://127.0.0.1:18080"), "coordinator base URL")
	fs.StringVar(&cfg.DatabaseURL, "database-url", env("DATABASE_URL", ""), "optional PostgreSQL URL for direct worker recovery is not used")
	fs.StringVar(&cfg.WorkerID, "worker-id", defaultID, "stable worker run identity")
	fs.StringVar(&cfg.StagingRoot, "staging-root", env("STAGING_ROOT", "./run/worker-tmp"), "temporary artifact root")
	fs.DurationVar(&cfg.PollInterval, "poll-interval", durationEnv("POLL_INTERVAL", 20*time.Millisecond), "poll interval")
	fs.StringVar(&cfg.LogPath, "log-path", env("WORKER_LOG_PATH", "./run/worker.log"), "structured log file")
	if err := fs.Parse(args); err != nil {
		return cfg, err
	}
	cfg.CoordinatorURL = strings.TrimRight(cfg.CoordinatorURL, "/")
	return cfg, validate(cfg.CoordinatorURL == "", "coordinator-url is required", cfg.WorkerID == "", "worker-id is required", cfg.StagingRoot == "", "staging-root is required", cfg.PollInterval <= 0, "poll-interval must be positive")
}

func env(key, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return fallback
}

func durationEnv(key string, fallback time.Duration) time.Duration {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return fallback
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		return fallback
	}
	return d
}

func validate(conds ...any) error {
	var problems []string
	for i := 0; i < len(conds); i += 2 {
		if conds[i].(bool) {
			problems = append(problems, conds[i+1].(string))
		}
	}
	if len(problems) > 0 {
		return errors.New(strings.Join(problems, "; "))
	}
	return nil
}

func RedactedDSN(dsn string) string {
	if dsn == "" {
		return ""
	}
	if u, err := urlParse(dsn); err == nil && u.User != nil {
		if _, ok := u.User.Password(); ok {
			u.User = urlUserPassword(u.User.Username(), "redacted")
		}
		return u.String()
	}
	return dsn
}

func urlParse(raw string) (*url.URL, error) { return url.Parse(raw) }

func urlUserPassword(username, password string) *url.Userinfo {
	return url.UserPassword(username, password)
}

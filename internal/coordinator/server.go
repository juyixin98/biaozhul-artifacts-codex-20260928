package coordinator

import (
	"log/slog"
	"net/http"
	"time"

	"opp300-speculative-batch/internal/artifacts"
	"opp300-speculative-batch/internal/store"
)

type Server struct {
	Store       *store.Store
	Artifacts   *artifacts.Store
	Staging     *stagingRegistry
	Logger      *slog.Logger
	Epoch       string
	BaseURL     string
	MaxArtifact int64
	Now         func() time.Time
}

func NewServer(dep *store.Store, artifactStore *artifacts.Store, logger *slog.Logger, epoch, baseURL string) *Server {
	return &Server{
		Store:       dep,
		Artifacts:   artifactStore,
		Staging:     newStagingRegistry(),
		Logger:      logger,
		Epoch:       epoch,
		BaseURL:     baseURL,
		MaxArtifact: 4 << 20,
		Now:         time.Now,
	}
}

func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.health)
	mux.HandleFunc("POST /v1/jobs", s.createJob)
	mux.HandleFunc("GET /v1/jobs/{id}", s.getJob)
	mux.HandleFunc("GET /v1/jobs/{id}/progress", s.getProgress)
	mux.HandleFunc("GET /v1/jobs/{id}/shards", s.listShards)
	mux.HandleFunc("GET /v1/jobs/{id}/attempts", s.listAttempts)
	mux.HandleFunc("GET /v1/jobs/{id}/events", s.listEvents)
	mux.HandleFunc("GET /v1/jobs/{id}/result", s.getResult)
	mux.HandleFunc("POST /v1/jobs/{id}/replay", s.replayEvents)

	mux.HandleFunc("POST /v1/attempts/{id}/start", s.startAttempt)
	mux.HandleFunc("POST /v1/attempts/{id}/heartbeats", s.heartbeat)
	mux.HandleFunc("POST /v1/attempts/{id}/fail", s.failAttempt)
	mux.HandleFunc("POST /v1/attempts/{id}/artifacts", s.stageArtifact)
	mux.HandleFunc("POST /v1/attempts/{id}/complete", s.completeAttempt)
	return requestLog(s.Logger, epochHeader(s.Epoch, mux))
}

func (s *Server) health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "epoch": s.Epoch})
}

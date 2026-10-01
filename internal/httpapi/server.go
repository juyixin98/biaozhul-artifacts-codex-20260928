package httpapi

import (
	"context"
	"errors"
	"net/http"
	"sync"

	"opp299/internal/config"
	"opp299/internal/engine"
	"opp299/internal/kernel"
	"opp299/internal/protocol"
	"opp299/internal/replay"
	"opp299/internal/source"
	"opp299/internal/storage"
)

type Server struct {
	cfg       config.Config
	store     storage.CheckpointStore
	mu        sync.Mutex
	workloads map[string]source.Workload
	results   map[string]RunResponse
}

func New(cfg config.Config, store storage.CheckpointStore) *Server {
	return &Server{
		cfg:       cfg,
		store:     store,
		workloads: map[string]source.Workload{},
		results:   map[string]RunResponse{},
	}
}

func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.health)
	mux.HandleFunc("PUT /v1/tasks/{taskID}/workload", s.putWorkload)
	mux.HandleFunc("GET /v1/tasks/{taskID}/workload", s.getWorkload)
	mux.HandleFunc("POST /v1/tasks/{taskID}/runs", s.run)
	mux.HandleFunc("GET /v1/tasks/{taskID}/checkpoints", s.listCheckpoints)
	mux.HandleFunc("GET /v1/tasks/{taskID}/checkpoints/latest", s.latestCheckpoint)
	mux.HandleFunc("GET /v1/tasks/{taskID}/checkpoints/{id}", s.getCheckpoint)
	mux.HandleFunc("POST /v1/tasks/{taskID}/recover", s.recover)
	return s.requestID(s.recoverLog(mux))
}

func (s *Server) health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok", "storage": s.cfg.Storage})
}

type RunRequest struct {
	Mode                     protocol.Mode `json:"mode"`
	BackpressureLimit        int           `json:"backpressure_limit"`
	MaxConcurrentCheckpoints int           `json:"max_concurrent_checkpoints"`
	FailCheckpointID         int64         `json:"fail_checkpoint_id"`
}

type RunResponse struct {
	RequestID string           `json:"request_id"`
	TaskID    string           `json:"task_id"`
	Result    engine.Result    `json:"result"`
	Steps     []string         `json:"steps"`
	Failure   *FailureResponse `json:"failure,omitempty"`
}

type FailureResponse struct {
	Category  kernel.ErrorCategory `json:"category"`
	Message   string               `json:"message"`
	Kind      string               `json:"kind,omitempty"`
	Uncertain bool                 `json:"uncertain"`
}

func (s *Server) putWorkload(w http.ResponseWriter, r *http.Request) {
	taskID := r.PathValue("taskID")
	var workload source.Workload
	if err := decode(r, &workload); err != nil {
		writeError(w, r, http.StatusBadRequest, kernel.CategoryProtocol, err)
		return
	}
	workload.TaskID = taskID
	if err := workload.Validate(); err != nil {
		writeError(w, r, http.StatusBadRequest, kernel.CategoryProtocol, err)
		return
	}
	s.mu.Lock()
	s.workloads[taskID] = workload
	delete(s.results, taskID)
	s.mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]any{"task_id": taskID, "events": countEvents(workload)})
}

func (s *Server) getWorkload(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	workload, ok := s.workloads[r.PathValue("taskID")]
	s.mu.Unlock()
	if !ok {
		writeError(w, r, http.StatusNotFound, kernel.CategoryNotFound, kernel.ErrCheckpointNotFound)
		return
	}
	writeJSON(w, http.StatusOK, workload)
}

func (s *Server) run(w http.ResponseWriter, r *http.Request) {
	taskID := r.PathValue("taskID")
	workload, err := s.requireWorkload(r.Context(), taskID)
	if err != nil {
		writeError(w, r, http.StatusNotFound, kernel.CategoryNotFound, err)
		return
	}
	req := RunRequest{
		Mode:                     protocol.Aligned,
		BackpressureLimit:        s.cfg.DefaultBackpressureLimit,
		MaxConcurrentCheckpoints: 1,
	}
	if err := decode(r, &req); err != nil {
		writeError(w, r, http.StatusBadRequest, kernel.CategoryProtocol, err)
		return
	}
	cfg := protocol.Config{
		TaskID:                   taskID,
		Mode:                     req.Mode,
		BackpressureLimit:        req.BackpressureLimit,
		MaxConcurrentCheckpoints: req.MaxConcurrentCheckpoints,
		FailCheckpointID:         req.FailCheckpointID,
	}
	if err := cfg.Validate(); err != nil {
		writeError(w, r, http.StatusBadRequest, kernel.CategoryProtocol, err)
		return
	}
	_, result := engine.RunWorkload(r.Context(), cfg, s.store, workload)
	response := RunResponse{
		RequestID: requestIDFromContext(r.Context()),
		TaskID:    taskID,
		Result:    result,
		Steps: []string{
			"accepted ordered synthetic events into bounded per-stream queues",
			"processed matching barrier boundaries according to mode",
			"stored complete checkpoint before committing barrier outputs",
		},
	}
	if result.Failure != nil {
		response.Failure = failureResponse(result.Failure)
	}
	s.mu.Lock()
	s.results[taskID] = response
	s.mu.Unlock()
	status := http.StatusOK
	if result.Failure != nil {
		status = http.StatusConflict
	}
	writeJSON(w, status, response)
}

func (s *Server) listCheckpoints(w http.ResponseWriter, r *http.Request) {
	cps, err := s.store.ListCheckpoints(r.Context(), r.PathValue("taskID"), 20)
	if err != nil {
		writeStorageError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, cps)
}

func (s *Server) latestCheckpoint(w http.ResponseWriter, r *http.Request) {
	cp, err := s.store.GetLatest(r.Context(), r.PathValue("taskID"))
	if err != nil {
		writeStorageError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, cp)
}

func (s *Server) getCheckpoint(w http.ResponseWriter, r *http.Request) {
	id, ok := parseID(w, r)
	if !ok {
		return
	}
	cp, err := s.store.GetCheckpoint(r.Context(), r.PathValue("taskID"), id)
	if err != nil {
		writeStorageError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, cp)
}

type RecoverResponse struct {
	RequestID string           `json:"request_id"`
	TaskID    string           `json:"task_id"`
	Replay    replay.Report    `json:"replay"`
	Result    engine.Result    `json:"result"`
	Steps     []string         `json:"steps"`
	Failure   *FailureResponse `json:"failure,omitempty"`
}

func (s *Server) recover(w http.ResponseWriter, r *http.Request) {
	taskID := r.PathValue("taskID")
	workload, err := s.requireWorkload(r.Context(), taskID)
	if err != nil {
		writeError(w, r, http.StatusNotFound, kernel.CategoryNotFound, err)
		return
	}
	var req RunRequest
	if err := decode(r, &req); err != nil {
		writeError(w, r, http.StatusBadRequest, kernel.CategoryProtocol, err)
		return
	}
	if req.BackpressureLimit == 0 {
		req.BackpressureLimit = s.cfg.DefaultBackpressureLimit
	}
	if req.MaxConcurrentCheckpoints == 0 {
		req.MaxConcurrentCheckpoints = 1
	}
	cp, err := s.store.GetLatest(r.Context(), taskID)
	if err != nil {
		writeStorageError(w, r, err)
		return
	}
	cfg := protocol.Config{
		TaskID:                   taskID,
		Mode:                     cp.Mode,
		BackpressureLimit:        req.BackpressureLimit,
		MaxConcurrentCheckpoints: req.MaxConcurrentCheckpoints,
	}
	if err := cfg.Validate(); err != nil {
		writeError(w, r, http.StatusBadRequest, kernel.CategoryProtocol, err)
		return
	}
	_, report, result := engine.RecoverAndRun(r.Context(), cfg, s.store, cp, workload)
	response := RecoverResponse{
		RequestID: requestIDFromContext(r.Context()),
		TaskID:    taskID,
		Replay:    report,
		Result:    result,
		Steps: []string{
			"loaded newest complete checkpoint",
			"applied only records explicitly carried in inflight",
			"resumed both synthetic streams strictly after barrier positions",
		},
	}
	if result.Failure != nil {
		response.Failure = failureResponse(result.Failure)
	}
	status := http.StatusOK
	if result.Failure != nil {
		status = http.StatusConflict
	}
	writeJSON(w, status, response)
}

func (s *Server) requireWorkload(_ context.Context, taskID string) (source.Workload, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	workload, ok := s.workloads[taskID]
	if ok {
		return workload, nil
	}
	path := "samples/two-speed.json"
	loaded, err := source.LoadWorkload(path)
	if err != nil {
		return workload, errors.New("workload has not been uploaded for this task")
	}
	if loaded.TaskID != taskID {
		return workload, errors.New("workload has not been uploaded for this task")
	}
	s.workloads[taskID] = loaded
	return loaded, nil
}

func failureResponse(err *kernel.ClassifiedError) *FailureResponse {
	return &FailureResponse{
		Category:  err.Category,
		Message:   err.Error(),
		Uncertain: err.Category == kernel.CategoryStorage,
	}
}

func countEvents(workload source.Workload) int {
	total := 0
	for _, stream := range workload.Streams {
		total += len(stream.Events)
	}
	return total
}

package api

import (
	"errors"
	"net/http"
	"sync"

	"checkpointlab/internal/kernel"
	"checkpointlab/internal/protocol"
	"checkpointlab/internal/storage"
)

type App struct {
	store storage.Store
	mu    sync.Mutex
	jobs  map[string]*job
}

type job struct {
	mode protocol.Mode
	cfg  kernel.Config
	op   *kernel.Operator
}

type Envelope struct {
	OK          bool         `json:"ok"`
	RequestID   string       `json:"request_id"`
	Data        any          `json:"data,omitempty"`
	Failure     *FailureBody `json:"failure,omitempty"`
	Uncertainty []string     `json:"uncertainty,omitempty"`
}

type FailureBody struct {
	Category  string `json:"category"`
	Message   string `json:"message"`
	Retriable bool   `json:"retriable"`
}

func New(store storage.Store) *App {
	return &App{store: store, jobs: map[string]*job{}}
}

func (a *App) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", a.health)
	mux.HandleFunc("POST /api/jobs", a.createJob)
	mux.HandleFunc("POST /api/jobs/{jobID}/events", a.postEvent)
	mux.HandleFunc("POST /api/jobs/{jobID}/restore", a.restore)
	mux.HandleFunc("GET /api/jobs/{jobID}/checkpoints", a.listCheckpoint)
	mux.HandleFunc("GET /api/jobs/{jobID}/outputs", a.listOutputs)
	mux.HandleFunc("GET /api/jobs/{jobID}/events-log", a.listEventLog)
	mux.HandleFunc("POST /api/scenarios/run", a.runScenario)
	return requestIDMiddleware(mux)
}

type createJobRequest struct {
	JobID         string        `json:"job_id"`
	Mode          protocol.Mode `json:"mode"`
	GateCapacity  int           `json:"gate_capacity"`
	InflightLimit int           `json:"inflight_limit"`
}

func (a *App) createJob(w http.ResponseWriter, r *http.Request) {
	var req createJobRequest
	if !decode(w, r, &req) {
		return
	}
	if req.GateCapacity == 0 {
		req.GateCapacity = 8
	}
	if req.InflightLimit == 0 {
		req.InflightLimit = 32
	}
	if err := a.store.CreateJob(r.Context(), req.JobID, req.Mode, nil); err != nil {
		writeError(w, r, http.StatusConflict, "JOB_CREATE_REJECTED", err.Error(), false)
		return
	}
	cfg := kernel.Config{
		JobID:         req.JobID,
		Mode:          req.Mode,
		GateCapacity:  req.GateCapacity,
		InflightLimit: req.InflightLimit,
	}
	op, err := kernel.NewOperator(cfg, a.store)
	if err != nil {
		writeError(w, r, http.StatusBadRequest, "KERNEL_CONFIG_REJECTED", err.Error(), false)
		return
	}
	a.mu.Lock()
	a.jobs[req.JobID] = &job{mode: req.Mode, cfg: cfg, op: op}
	a.mu.Unlock()
	writeJSON(w, http.StatusCreated, Envelope{OK: true, RequestID: requestID(r), Data: cfg})
}

type eventRequest struct {
	Record  *protocol.Record  `json:"record,omitempty"`
	Barrier *protocol.Barrier `json:"barrier,omitempty"`
}

func (a *App) postEvent(w http.ResponseWriter, r *http.Request) {
	j, ok := a.requireJob(w, r)
	if !ok {
		return
	}
	var req eventRequest
	if !decode(w, r, &req) {
		return
	}
	var event protocol.Event
	switch {
	case req.Record != nil && req.Barrier != nil:
		writeError(w, r, http.StatusBadRequest, "VALIDATION_REJECTED", "send exactly one record or barrier", false)
		return
	case req.Record != nil:
		event = protocol.RecordEvent(*req.Record)
	case req.Barrier != nil:
		event = protocol.BarrierEvent(*req.Barrier)
	default:
		writeError(w, r, http.StatusBadRequest, "VALIDATION_REJECTED", "record or barrier is required", false)
		return
	}
	result, err := j.op.Process(r.Context(), event, requestID(r))
	if err != nil {
		status := http.StatusBadRequest
		var typed *protocol.TypedError
		if errors.As(err, &typed) && (typed.Category == protocol.CategoryStorageFailed ||
			typed.Category == protocol.CategoryUncertainStorage) {
			status = http.StatusServiceUnavailable
		}
		writeError(w, r, status, errorCategory(err), err.Error(), isRetriable(err))
		return
	}
	writeJSON(w, http.StatusOK, Envelope{OK: true, RequestID: requestID(r), Data: result})
}

func (a *App) restore(w http.ResponseWriter, r *http.Request) {
	j, ok := a.requireJob(w, r)
	if !ok {
		return
	}
	point, err := j.op.Restore(r.Context())
	if err != nil {
		writeError(w, r, http.StatusServiceUnavailable, "RESTORE_FAILED", err.Error(), true)
		return
	}
	writeJSON(w, http.StatusOK, Envelope{OK: true, RequestID: requestID(r), Data: point})
}

func (a *App) listCheckpoint(w http.ResponseWriter, r *http.Request) {
	jobID := r.PathValue("jobID")
	if active, ok, _ := a.store.ActiveCheckpoint(r.Context(), jobID); ok {
		writeJSON(w, http.StatusOK, Envelope{OK: true, RequestID: requestID(r), Data: active})
		return
	}
	latest, ok, err := a.store.LatestSealed(r.Context(), jobID)
	if err != nil {
		writeError(w, r, http.StatusInternalServerError, "STORAGE_READ_FAILED", err.Error(), true)
		return
	}
	if !ok {
		writeError(w, r, http.StatusNotFound, "CHECKPOINT_NOT_FOUND", "no checkpoint exists", false)
		return
	}
	writeJSON(w, http.StatusOK, Envelope{OK: true, RequestID: requestID(r), Data: latest})
}

func (a *App) listOutputs(w http.ResponseWriter, r *http.Request) {
	jobID := r.PathValue("jobID")
	committedOnly := r.URL.Query().Get("committed") != "false"
	outs, err := a.store.ListOutputs(r.Context(), jobID, committedOnly)
	if err != nil {
		writeError(w, r, http.StatusInternalServerError, "STORAGE_READ_FAILED", err.Error(), true)
		return
	}
	writeJSON(w, http.StatusOK, Envelope{OK: true, RequestID: requestID(r), Data: outs})
}

func (a *App) listEventLog(w http.ResponseWriter, r *http.Request) {
	jobID := r.PathValue("jobID")
	entries, err := a.store.ListEvents(r.Context(), jobID, 100)
	if err != nil {
		writeError(w, r, http.StatusInternalServerError, "STORAGE_READ_FAILED", err.Error(), true)
		return
	}
	writeJSON(w, http.StatusOK, Envelope{OK: true, RequestID: requestID(r), Data: entries})
}

func (a *App) health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, Envelope{OK: true, RequestID: requestID(r), Data: map[string]string{"status": "ok"}})
}

func (a *App) Close() {
	a.mu.Lock()
	defer a.mu.Unlock()
	for _, j := range a.jobs {
		j.op.Close()
	}
	_ = a.store.Close()
}

func (a *App) requireJob(w http.ResponseWriter, r *http.Request) (*job, bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	j, ok := a.jobs[r.PathValue("jobID")]
	if !ok {
		writeError(w, r, http.StatusNotFound, "JOB_NOT_FOUND", "create the job before sending events", false)
		return nil, false
	}
	return j, true
}

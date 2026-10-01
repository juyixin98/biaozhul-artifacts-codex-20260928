//go:build integration

package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	"opp300-speculative-batch/internal/artifacts"
	"opp300-speculative-batch/internal/coordinator"
	"opp300-speculative-batch/internal/idgen"
	"opp300-speculative-batch/internal/participant"
	"opp300-speculative-batch/internal/protocol"
	"opp300-speculative-batch/internal/scheduler"
	"opp300-speculative-batch/internal/store"
	"opp300-speculative-batch/internal/worker"
)

type harness struct {
	t           *testing.T
	store       *store.Store
	artifacts   *artifacts.Store
	logger      *slog.Logger
	artifactDir string
	epoch       string

	coordinatorURL  string
	workerURL       string
	httpServer      *http.Server
	workerServer    *http.Server
	cancel          context.CancelFunc
	workerControl   *worker.Server
	scheduler       *scheduler.Scheduler
	schedulerCancel context.CancelFunc
	schedulerDone   chan struct{}
	listener        net.Listener
	workerListener  net.Listener
}

func newHarness(t *testing.T, dbURL string) *harness {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())

	st, closeStore, err := store.New(ctx, dbURL)
	if err != nil {
		cancel()
		t.Fatalf("connect store: %v", err)
	}
	t.Cleanup(func() {
		closeStore()
	})
	if err := st.Migrate(ctx, filepath.Join("..", "migrations")); err != nil {
		cancel()
		t.Fatalf("migrate: %v", err)
	}

	artifactDir := filepath.Join(t.TempDir(), "artifacts")
	artifactStore, err := artifacts.NewStore(artifactDir)
	if err != nil {
		cancel()
		t.Fatalf("artifact store: %v", err)
	}

	logger := slog.New(slog.NewTextHandler(&testLogWriter{t: t}, &slog.HandlerOptions{Level: slog.LevelInfo}))
	epoch, err := idgen.New("epoch")
	if err != nil {
		cancel()
		t.Fatal(err)
	}

	coordListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		cancel()
		t.Fatal(err)
	}
	workerListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		_ = coordListener.Close()
		cancel()
		t.Fatal(err)
	}

	h := &harness{
		t:              t,
		store:          st,
		artifacts:      artifactStore,
		logger:         logger,
		artifactDir:    artifactDir,
		epoch:          epoch,
		listener:       coordListener,
		workerListener: workerListener,
		cancel:         cancel,
		coordinatorURL: "http://" + coordListener.Addr().String(),
		workerURL:      "http://" + workerListener.Addr().String(),
		schedulerDone:  make(chan struct{}),
	}

	coord := coordinator.NewServer(st, artifactStore, logger.With("component", "coordinator"), epoch, h.coordinatorURL)
	h.httpServer = &http.Server{Handler: coord.Routes()}
	kernel := &worker.Kernel{Client: participant.NewClient()}
	h.workerControl = worker.NewServer(kernel, logger.With("component", "worker"))
	h.workerServer = &http.Server{Handler: h.workerControl.Routes()}

	go func() {
		if err := h.httpServer.Serve(coordListener); err != nil && err != http.ErrServerClosed {
			t.Errorf("coordinator server: %v", err)
		}
	}()
	go func() {
		if err := h.workerServer.Serve(workerListener); err != nil && err != http.ErrServerClosed {
			t.Errorf("worker server: %v", err)
		}
	}()

	h.scheduler = &scheduler.Scheduler{
		Store:     st,
		WorkerURL: h.workerURL,
		BaseURL:   h.coordinatorURL,
		Epoch:     epoch,
		Poll:      15 * time.Millisecond,
		LeasePoll: 15 * time.Millisecond,
		Logger:    logger.With("component", "scheduler"),
	}
	schedCtx, schedCancel := context.WithCancel(ctx)
	h.schedulerCancel = schedCancel
	go func() {
		defer close(h.schedulerDone)
		h.scheduler.Run(schedCtx)
	}()

	t.Cleanup(h.close)
	h.waitReady(ctx)
	return h
}

func (h *harness) close() {
	h.schedulerCancel()
	<-h.schedulerDone
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	_ = h.httpServer.Shutdown(shutdownCtx)
	_ = h.workerServer.Shutdown(shutdownCtx)
	h.workerControl.Shutdown()
	h.cancel()
}

func (h *harness) waitReady(ctx context.Context) {
	for _, url := range []string{h.coordinatorURL + "/healthz", h.workerURL + "/healthz"} {
		for i := 0; i < 50; i++ {
			request, _ := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
			response, err := http.DefaultClient.Do(request)
			if err == nil {
				_ = response.Body.Close()
				if response.StatusCode == 200 {
					break
				}
			}
			time.Sleep(20 * time.Millisecond)
			if i == 49 {
				h.t.Fatalf("service never became ready: %s", url)
			}
		}
	}
}

func (h *harness) restartCoordinator(ctx context.Context) {
	oldEpoch := h.epoch
	h.schedulerCancel()
	<-h.schedulerDone
	shutdownCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
	if err := h.httpServer.Shutdown(shutdownCtx); err != nil {
		h.t.Fatalf("shutdown coordinator: %v", err)
	}
	cancel()
	listener, err := net.Listen("tcp", h.listener.Addr().String())
	if err != nil {
		h.t.Fatalf("rebind coordinator address: %v", err)
	}
	h.listener = listener

	epoch, err := idgen.New("epoch")
	if err != nil {
		h.t.Fatal(err)
	}
	h.epoch = epoch
	coord := coordinator.NewServer(h.store, h.artifacts, h.logger.With("component", "coordinator-restarted"), epoch, h.coordinatorURL)
	h.httpServer = &http.Server{Handler: coord.Routes()}
	go func() {
		if err := h.httpServer.Serve(listener); err != nil && err != http.ErrServerClosed {
			h.t.Errorf("restarted coordinator server: %v", err)
		}
	}()

	h.schedulerDone = make(chan struct{})
	schedCtx, schedCancel := context.WithCancel(ctx)
	h.schedulerCancel = schedCancel
	h.scheduler = &scheduler.Scheduler{
		Store:     h.store,
		WorkerURL: h.workerURL,
		BaseURL:   h.coordinatorURL,
		Epoch:     epoch,
		Poll:      15 * time.Millisecond,
		LeasePoll: 15 * time.Millisecond,
		Logger:    h.logger.With("component", "scheduler-restarted"),
	}
	go func() {
		defer close(h.schedulerDone)
		h.scheduler.Run(schedCtx)
	}()
	h.waitReady(ctx)
	h.t.Logf("coordinator restarted old_epoch=%s new_epoch=%s", oldEpoch, epoch)
}

type testLogWriter struct{ t *testing.T }

func (w *testLogWriter) Write(p []byte) (int, error) {
	w.t.Log(stringsTrim(string(p)))
	return len(p), nil
}

func stringsTrim(value string) string {
	for len(value) > 0 && (value[len(value)-1] == '\n' || value[len(value)-1] == '\r') {
		value = value[:len(value)-1]
	}
	return value
}

func (h *harness) submit(ctx context.Context, spec protocol.JobSpec) protocol.Job {
	h.t.Helper()
	response := h.doJSON(ctx, http.MethodPost, h.coordinatorURL+"/v1/jobs", spec)
	if response.StatusCode != http.StatusCreated {
		body, _ := io.ReadAll(response.Body)
		response.Body.Close()
		h.t.Fatalf("submit status=%d body=%s", response.StatusCode, string(body))
	}
	defer response.Body.Close()
	var envelope struct {
		Job protocol.Job `json:"job"`
	}
	if err := json.NewDecoder(response.Body).Decode(&envelope); err != nil {
		h.t.Fatal(err)
	}
	return envelope.Job
}

func (h *harness) doJSON(ctx context.Context, method, url string, value any) *http.Response {
	h.t.Helper()
	var body io.Reader
	if value != nil {
		payload, err := json.Marshal(value)
		if err != nil {
			h.t.Fatal(err)
		}
		body = bytes.NewReader(payload)
	}
	request, err := http.NewRequestWithContext(ctx, method, url, body)
	if err != nil {
		h.t.Fatal(err)
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		h.t.Fatal(err)
	}
	return response
}

func (h *harness) getJSON(ctx context.Context, url string, dst any) {
	h.t.Helper()
	response := h.doJSON(ctx, http.MethodGet, url, nil)
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(response.Body)
		h.t.Fatalf("GET %s status=%d body=%s", url, response.StatusCode, string(body))
	}
	if err := json.NewDecoder(response.Body).Decode(dst); err != nil {
		h.t.Fatal(err)
	}
}

func (h *harness) waitJobState(ctx context.Context, jobID, state string) protocol.Job {
	h.t.Helper()
	deadline, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	for {
		var job protocol.Job
		h.getJSON(deadline, fmt.Sprintf("%s/v1/jobs/%s", h.coordinatorURL, jobID), &job)
		if job.State == state {
			return job
		}
		if (state == protocol.JobSucceeded && job.State == protocol.JobFailed) ||
			(state == protocol.JobFailed && job.State == protocol.JobSucceeded) {
			h.t.Fatalf("job reached unexpected terminal state %#v waiting for %#v", job.State, state)
		}
		select {
		case <-deadline.Done():
			h.t.Fatalf("job %s did not reach %s, last=%#v", jobID, state, job)
		case <-time.After(15 * time.Millisecond):
		}
	}
}

func (h *harness) events(ctx context.Context, jobID string) []protocol.Event {
	var envelope struct {
		Events []protocol.Event `json:"events"`
	}
	h.getJSON(ctx, fmt.Sprintf("%s/v1/jobs/%s/events?limit=1000", h.coordinatorURL, jobID), &envelope)
	return envelope.Events
}

func (h *harness) attempts(ctx context.Context, jobID string) []protocol.Attempt {
	var envelope struct {
		Attempts []protocol.Attempt `json:"attempts"`
	}
	h.getJSON(ctx, fmt.Sprintf("%s/v1/jobs/%s/attempts", h.coordinatorURL, jobID), &envelope)
	return envelope.Attempts
}

func (h *harness) result(ctx context.Context, jobID string) protocol.MergeResult {
	var merge protocol.MergeResult
	h.getJSON(ctx, fmt.Sprintf("%s/v1/jobs/%s/result", h.coordinatorURL, jobID), &merge)
	return merge
}

func databaseURL(t *testing.T) string {
	t.Helper()
	url := os.Getenv("TEST_DATABASE_URL")
	if url == "" {
		url = "postgres:///opp300_test?host=/var/run/postgresql&sslmode=disable"
	}
	return url
}

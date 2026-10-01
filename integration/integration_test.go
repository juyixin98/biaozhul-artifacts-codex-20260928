package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"specbatch/internal/model"
	"specbatch/internal/store"
)

const dbURL = "postgres://admin:specbatch@127.0.0.1:5432/specbatch?sslmode=disable"

type env struct {
	t       *testing.T
	root    string
	coord   string
	worker  string
	schema  string
	baseDSN string
}

type process struct {
	cmd *exec.Cmd
}

func setupEnv(t *testing.T) *env {
	t.Helper()
	root := t.TempDir()
	ctx := context.Background()
	admin, err := store.Open(ctx, dbURL)
	if err != nil {
		t.Skipf("PostgreSQL unavailable: %v", err)
	}
	schema := fmt.Sprintf("it_%d", time.Now().UnixNano())
	if _, err := admin.DB().ExecContext(ctx, `DROP SCHEMA IF EXISTS `+schema+` CASCADE; CREATE SCHEMA `+schema); err != nil {
		admin.Close()
		t.Skipf("cannot create isolated schema: %v", err)
	}
	admin.Close()

	coord := filepath.Join(root, "coordinator")
	worker := filepath.Join(root, "worker")
	build(t, coord, "./cmd/specbatch-coordinator")
	build(t, worker, "./cmd/specbatch-worker")
	u, _ := url.Parse(dbURL)
	q := u.Query()
	q.Set("search_path", schema)
	u.RawQuery = q.Encode()
	return &env{t: t, root: root, coord: coord, worker: worker, schema: schema, baseDSN: u.String()}
}

func build(t *testing.T, output, pkg string) {
	t.Helper()
	cmd := exec.Command("go", "build", "-o", output, pkg)
	cmd.Dir, _ = filepath.Abs("..")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build %s: %v\n%s", pkg, err, out)
	}
}

func freePort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

func (e *env) startCoordinator(t *testing.T, port int) *process {
	t.Helper()
	logPath := filepath.Join(e.root, fmt.Sprintf("coordinator-%d.log", port))
	args := []string{
		"-http-addr", fmt.Sprintf("127.0.0.1:%d", port),
		"-database-url", e.baseDSN,
		"-artifact-dir", filepath.Join(e.root, "artifacts"),
		"-schedule-tick", "10ms",
		"-heartbeat-ttl", "100ms",
		"-log-path", logPath,
	}
	cmd := exec.Command(e.coord, args...)
	cmd.Dir = e.root
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	p := &process{cmd: cmd}
	t.Cleanup(func() { _ = p.Stop() })
	waitHTTP(t, fmt.Sprintf("http://127.0.0.1:%d/healthz", port), 5*time.Second)
	return p
}

func (e *env) startWorker(t *testing.T, port int, id string) *process {
	t.Helper()
	cmd := exec.Command(e.worker,
		"-coordinator-url", fmt.Sprintf("http://127.0.0.1:%d", port),
		"-worker-id", id,
		"-staging-root", filepath.Join(e.root, "staging", id),
		"-poll-interval", "10ms",
		"-log-path", filepath.Join(e.root, "worker-"+id+".log"),
	)
	cmd.Dir = e.root
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	p := &process{cmd: cmd}
	t.Cleanup(func() { _ = p.Stop() })
	return p
}

func (p *process) Stop() error {
	if p == nil || p.cmd == nil || p.cmd.ProcessState != nil {
		return nil
	}
	_ = p.cmd.Process.Signal(os.Interrupt)
	done := make(chan error, 1)
	go func() { done <- p.cmd.Wait() }()
	select {
	case <-done:
		return nil
	case <-time.After(2 * time.Second):
		_ = p.cmd.Process.Kill()
		<-done
		return nil
	}
}

func (p *process) KillForRestart(t *testing.T) {
	t.Helper()
	if err := p.cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	_ = p.cmd.Wait()
	p.cmd = nil
}

func waitHTTP(t *testing.T, addr string, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	client := &http.Client{Timeout: 500 * time.Millisecond}
	for time.Now().Before(deadline) {
		resp, err := client.Get(addr)
		if err == nil {
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			if resp.StatusCode == 200 {
				return
			}
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("service did not become ready: %s", addr)
}

func postJSON(t *testing.T, base string, path string, body any) []byte {
	t.Helper()
	raw, _ := json.Marshal(body)
	resp, err := http.Post(base+path, "application/json", bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 300 {
		t.Fatalf("POST %s status=%d body=%s", path, resp.StatusCode, string(data))
	}
	return data
}

func getJSON(t *testing.T, base, path string, out any) {
	t.Helper()
	resp, err := http.Get(base + path)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 300 {
		t.Fatalf("GET %s status=%d body=%s", path, resp.StatusCode, string(data))
	}
	if err := json.Unmarshal(data, out); err != nil {
		t.Fatal(err)
	}
}

func loadFixture(t *testing.T, path, jobID string) model.Manifest {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", path))
	if err != nil {
		t.Fatal(err)
	}
	var req struct {
		Manifest model.Manifest `json:"manifest"`
	}
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatal(err)
	}
	req.Manifest.JobID = jobID
	return req.Manifest
}

func loadExpected(t *testing.T, path, jobID string) model.MergeDocument {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", path))
	if err != nil {
		t.Fatal(err)
	}
	var doc model.MergeDocument
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	doc.JobID = jobID
	return doc
}

func waitJob(t *testing.T, base, jobID string, timeout time.Duration) store.JobStatusResponse {
	t.Helper()
	deadline := time.Now().Add(timeout)
	var status store.JobStatusResponse
	for time.Now().Before(deadline) {
		getJSON(t, base, "/v1/jobs/"+jobID+"?attempts=1", &status)
		if status.Job.Status == model.JobSucceeded || status.Job.Status == model.JobFailed {
			return status
		}
		time.Sleep(25 * time.Millisecond)
	}
	t.Fatalf("job %s did not finish; last=%+v", jobID, status)
	return status
}

func assertNoFileUnder(t *testing.T, root string, pred func(string) bool) {
	t.Helper()
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			if os.IsNotExist(err) {
				return nil
			}
			return err
		}
		if !info.IsDir() && pred(path) {
			t.Fatalf("temporary artifact was retained: %s", path)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

func readFile(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func assertContainsAll(t *testing.T, text string, needles ...string) {
	t.Helper()
	for _, needle := range needles {
		if !strings.Contains(text, needle) {
			t.Fatalf("log %s missing %q", callerFileHint(), needle)
		}
	}
}

func callerFileHint() string { return "output" }

func itoa(v int) string { return strconv.Itoa(v) }

func uniqueSuffix() string { return strconv.FormatInt(time.Now().UnixNano(), 36) }

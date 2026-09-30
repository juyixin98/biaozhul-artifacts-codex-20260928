package service

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"log"
	"net/http"
	"time"

	"github.com/jackc/pgx/v5"

	"opp297/topk/internal/domain"
	"opp297/topk/internal/kernel"
	"opp297/topk/internal/participant"
	"opp297/topk/internal/storage"
)

type Coordinator struct {
	Store       *storage.Store
	ShardURLs   map[int]string
	Client      kernel.Reader
	ShardIDs    []int
	PageSize    int
	Timeout     time.Duration
	BoundSecret []byte
	Logger      *log.Logger
}

type requestReader struct {
	client    *participant.MultiClient
	requestID string
	fault     string
}

func (r *requestReader) ReadPage(ctx context.Context, req domain.PageRequest) (domain.PageResponse, error) {
	ctx = withRequestID(ctx, r.requestID)
	if r.fault != "" {
		ctx = withTestFault(ctx, r.fault, req.ShardID)
	}
	return r.client.ReadPage(ctx, req)
}

type requestIDKey struct{}
type testFaultKey struct{}

func withRequestID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, requestIDKey{}, id)
}

func withTestFault(ctx context.Context, spec string, shardID int) context.Context {
	fault, target, ok := parseFaultSpec(spec)
	if !ok || (target >= 0 && target != shardID) {
		return ctx
	}
	return context.WithValue(ctx, testFaultKey{}, fault)
}

func parseFaultSpec(spec string) (string, int, bool) {
	for i := 0; i < len(spec); i++ {
		if spec[i] == ':' {
			var shard int
			for _, r := range spec[i+1:] {
				if r < '0' || r > '9' {
					return "", 0, false
				}
				shard = shard*10 + int(r-'0')
			}
			return spec[:i], shard, true
		}
	}
	return spec, -1, true
}

func requestIDFromContext(ctx context.Context) string {
	if v, ok := ctx.Value(requestIDKey{}).(string); ok {
		return v
	}
	return ""
}

func testFaultFromContext(ctx context.Context) string {
	if v, ok := ctx.Value(testFaultKey{}).(string); ok {
		return v
	}
	return ""
}

type contextTransport struct {
	base http.RoundTripper
}

func (t contextTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if id := requestIDFromContext(req.Context()); id != "" {
		req.Header.Set("X-Request-ID", id)
	}
	if fault := testFaultFromContext(req.Context()); fault != "" {
		req.Header.Set("X-Test-Fault", fault)
	}
	base := t.base
	if base == nil {
		base = http.DefaultTransport
	}
	return base.RoundTrip(req)
}

func NewRequestClient(urls map[int]string, requestID, faultSpec string) kernel.Reader {
	multi := participant.NewMultiClient(urls)
	for _, client := range multi.Clients {
		client.HTTP = &http.Client{Transport: contextTransport{}}
	}
	return &requestReader{client: multi, requestID: requestID, fault: faultSpec}
}

func NewRequestID() (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	return "req_" + hex.EncodeToString(b[:]), nil
}

func (c *Coordinator) Execute(ctx context.Context, requestID string, gen domain.GenerationID, k int, faultSpec string) (kernel.Result, error) {
	exists, err := c.Store.GenerationExists(ctx, gen)
	if err != nil {
		return kernel.Result{}, err
	}
	if !exists {
		return kernel.Result{}, ErrUnknownGeneration
	}

	started := time.Now().UTC()
	reader := c.Client
	if reader == nil {
		reader = NewRequestClient(c.ShardURLs, requestID, faultSpec)
	}

	result, err := kernel.Execute(ctx, gen, kernel.Options{
		ShardIDs:    c.ShardIDs,
		K:           k,
		PageSize:    c.PageSize,
		Timeout:     c.Timeout,
		BoundSecret: c.BoundSecret,
	}, reader)
	if err != nil {
		return kernel.Result{}, err
	}

	completed := time.Now().UTC()
	pq := storage.PersistedQuery{
		RequestID:     requestID,
		Generation:    gen,
		K:             k,
		Exact:         result.Exact,
		Complete:      result.Complete,
		Threshold:     lastThreshold(result.Events),
		Rounds:        result.Rounds,
		TotalRowsRead: result.TotalRows,
		StartedAt:     started,
		CompletedAt:   completed,
		Keys:          result.Keys,
		Statuses:      convertStatuses(result.Statuses),
		Events:        convertEvents(result.Events),
	}
	if result.Failure != nil {
		pq.FailureClass = string(result.Failure.Class)
		pq.FailureMessage = result.Failure.Error()
	}
	if err := c.Store.SaveQuery(ctx, pq); err != nil {
		return result, err
	}
	c.Logger.Printf("request_id=%s gen=%s exact=%t complete=%t failure=%s rounds=%d rows=%d result_count=%d",
		requestID, redactGeneration(gen), result.Exact, result.Complete, failureClass(result), result.Rounds, result.TotalRows, len(result.Keys))
	return result, nil
}

func (c *Coordinator) Replay(ctx context.Context, requestID string) (storage.Replay, error) {
	replay, err := c.Store.GetReplay(ctx, requestID)
	if errors.Is(err, pgx.ErrNoRows) {
		return replay, ErrUnknownRequest
	}
	return replay, err
}

func convertStatuses(in []kernel.ShardStatus) []storage.PersistedStatus {
	out := make([]storage.PersistedStatus, 0, len(in))
	for _, st := range in {
		out = append(out, storage.PersistedStatus{
			ID: st.ID, Exhausted: st.Exhausted, Active: st.Active,
			UpperBound: st.UpperBound, Cursor: st.Cursor, Pages: st.Pages, RowsRead: st.RowsRead,
		})
	}
	return out
}

func convertEvents(in []kernel.Event) []storage.PersistedEvent {
	out := make([]storage.PersistedEvent, 0, len(in))
	for _, ev := range in {
		pev := storage.PersistedEvent{
			Kind: string(ev.Kind), ShardID: ev.ShardID, Round: ev.Round,
			Cursor: ev.Cursor, UpperBound: ev.UpperBound, Items: ev.Items,
			Threshold: ev.Threshold, Reason: ev.Reason, RowsRead: ev.RowsRead, At: ev.At,
		}
		if ev.Failure != nil {
			pev.FailureClass = string(ev.Failure.Class)
		}
		out = append(out, pev)
	}
	return out
}

func lastThreshold(events []kernel.Event) *domain.Key {
	for i := len(events) - 1; i >= 0; i-- {
		if events[i].Threshold != nil {
			k := *events[i].Threshold
			return &k
		}
	}
	return nil
}

func failureClass(r kernel.Result) string {
	if r.Failure == nil {
		return "-"
	}
	return string(r.Failure.Class)
}

func redactGeneration(gen domain.GenerationID) string {
	text := string(gen)
	if len(text) <= 8 {
		if text == "" {
			return "-"
		}
		return text[:1] + "***"
	}
	return text[:4] + "***" + text[len(text)-4:]
}

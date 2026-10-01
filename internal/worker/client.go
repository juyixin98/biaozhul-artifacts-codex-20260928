package worker

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"time"

	"specbatch/internal/model"
)

type Client struct {
	BaseURL string
	HTTP    *http.Client
}

func NewClient(baseURL string) *Client {
	return &Client{BaseURL: baseURL, HTTP: &http.Client{Timeout: 5 * time.Second}}
}

func (c *Client) doJSON(ctx context.Context, method, path string, req any, out any) (int, []byte, error) {
	var body io.Reader
	if req != nil {
		raw, err := json.Marshal(req)
		if err != nil {
			return 0, nil, err
		}
		body = bytes.NewReader(raw)
	}
	httpReq, err := http.NewRequestWithContext(ctx, method, c.BaseURL+path, body)
	if err != nil {
		return 0, nil, err
	}
	if req != nil {
		httpReq.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.HTTP.Do(httpReq)
	if err != nil {
		return 0, nil, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return resp.StatusCode, nil, err
	}
	if out != nil && len(raw) > 0 {
		if err := json.Unmarshal(raw, out); err != nil {
			return resp.StatusCode, raw, err
		}
	}
	if resp.StatusCode >= 300 {
		return resp.StatusCode, raw, fmt.Errorf("http %d: %s", resp.StatusCode, string(raw))
	}
	return resp.StatusCode, raw, nil
}

type pollResponse struct {
	Lease model.Lease `json:"lease"`
	Epoch int64       `json:"epoch"`
}

func (c *Client) Poll(ctx context.Context, workerID string) (model.Lease, bool, error) {
	status, raw, err := c.doJSON(ctx, http.MethodPost, "/internal/v1/poll", map[string]string{"worker_id": workerID}, nil)
	if err != nil {
		if status == http.StatusNoContent || isTimeout(raw) {
			return model.Lease{}, false, nil
		}
		return model.Lease{}, false, err
	}
	if status == http.StatusNoContent {
		return model.Lease{}, false, nil
	}
	var resp pollResponse
	if err := json.Unmarshal(raw, &resp); err != nil {
		return model.Lease{}, false, err
	}
	return resp.Lease, true, nil
}

func (c *Client) Heartbeat(ctx context.Context, token, workerID string, progress float64) error {
	_, _, err := c.doJSON(ctx, http.MethodPost, "/internal/v1/heartbeat", map[string]any{
		"token": token, "worker_id": workerID, "progress": progress,
	}, nil)
	return err
}

func (c *Client) Fail(ctx context.Context, token, workerID, category, message string) error {
	_, _, err := c.doJSON(ctx, http.MethodPost, "/internal/v1/fail", map[string]string{
		"token": token, "worker_id": workerID, "category": category, "message": message,
	}, nil)
	return err
}

func (c *Client) Complete(ctx context.Context, token, workerID string, artifact []byte) error {
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	_ = mw.WriteField("token", token)
	_ = mw.WriteField("worker_id", workerID)
	part, err := mw.CreateFormFile("artifact", "artifact.json")
	if err != nil {
		return err
	}
	if _, err := part.Write(artifact); err != nil {
		return err
	}
	if err := mw.Close(); err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/internal/v1/complete", &buf)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", mw.FormDataContentType())
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if resp.StatusCode >= 300 {
		return fmt.Errorf("http %d: %s", resp.StatusCode, string(raw))
	}
	return nil
}

func isTimeout(raw []byte) bool {
	return false
}

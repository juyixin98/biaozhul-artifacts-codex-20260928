package coordinator

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"topk-demo/internal/protocol"
	"topk-demo/internal/topk"
)

type HTTPShardClient struct {
	baseURL string
	http    *http.Client
}

func NewHTTPShardClient(baseURL string) *HTTPShardClient {
	return &HTTPShardClient{
		baseURL: strings.TrimRight(baseURL, "/"),
		http:    &http.Client{Timeout: 10 * time.Second},
	}
}

func (c *HTTPShardClient) Bound(ctx context.Context, req protocol.BoundRequest) (*protocol.BoundResponse, error) {
	var out protocol.BoundResponse
	if err := c.post(ctx, "/v1/shards/bound", req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *HTTPShardClient) Scan(ctx context.Context, req protocol.ScanRequest) (*protocol.ScanResponse, error) {
	var out protocol.ScanResponse
	if err := c.post(ctx, "/v1/shards/scan", req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *HTTPShardClient) post(ctx context.Context, path string, body any, out any) error {
	raw, err := json.Marshal(body)
	if err != nil {
		return &topk.ShardCallError{Category: protocol.FailureInvalidPayload, Message: err.Error()}
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(raw))
	if err != nil {
		return &topk.ShardCallError{Category: protocol.FailureNetwork, Message: err.Error()}
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if rid, ok := ctx.Value(requestIDKey{}).(string); ok && rid != "" {
		httpReq.Header.Set("X-Request-ID", rid)
	}
	resp, err := c.http.Do(httpReq)
	if err != nil {
		category := protocol.FailureNetwork
		if ctx.Err() == context.DeadlineExceeded {
			category = protocol.FailureTimeout
		}
		return &topk.ShardCallError{Category: category, Message: err.Error()}
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		var errBody struct {
			Error    string `json:"error"`
			Category string `json:"category"`
		}
		_ = json.Unmarshal(data, &errBody)
		category := statusCategory(resp.StatusCode)
		if errBody.Category == protocol.FailureBadGeneration {
			category = protocol.FailureBadGeneration
		}
		message := errBody.Error
		if message == "" {
			message = fmt.Sprintf("unexpected HTTP status %d", resp.StatusCode)
		}
		return &topk.ShardCallError{Category: category, Message: message}
	}
	if err := json.Unmarshal(data, out); err != nil {
		return &topk.ShardCallError{Category: protocol.FailureInvalidPayload, Message: "decode shard response: " + err.Error()}
	}
	return nil
}

func statusCategory(status int) string {
	if status == http.StatusConflict {
		return protocol.FailureBadGeneration
	}
	if status >= 400 && status < 600 {
		return protocol.FailureHTTPStatus
	}
	return protocol.FailureNetwork
}

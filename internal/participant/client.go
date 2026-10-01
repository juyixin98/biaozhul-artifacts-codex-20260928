package participant

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

type Client struct {
	HTTP *http.Client
}

func NewClient() *Client {
	return &Client{HTTP: &http.Client{Timeout: 3 * time.Second}}
}

func (c *Client) Start(ctx context.Context, baseURL, attemptID string, req StartRequest) error {
	return c.postJSON(ctx, http.MethodPost, baseURL+"/v1/attempts/"+attemptID+"/start", req, nil)
}

func (c *Client) Heartbeat(ctx context.Context, baseURL, attemptID, epoch string) error {
	return c.postJSON(ctx, http.MethodPost, baseURL+"/v1/attempts/"+attemptID+"/heartbeats",
		map[string]string{"epoch": epoch}, nil)
}

func (c *Client) Fail(ctx context.Context, baseURL, attemptID string, req FailureRequest) error {
	return c.postJSON(ctx, http.MethodPost, baseURL+"/v1/attempts/"+attemptID+"/fail", req, nil)
}

func (c *Client) StageArtifact(ctx context.Context, baseURL, attemptID, epoch, checksum string, body []byte) (StageResponse, error) {
	endpoint, err := url.Parse(strings.TrimRight(baseURL, "/") + "/v1/attempts/" + attemptID + "/artifacts")
	if err != nil {
		return StageResponse{}, err
	}
	query := endpoint.Query()
	query.Set("checksum_sha256", checksum)
	query.Set("epoch", epoch)
	endpoint.RawQuery = query.Encode()
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint.String(), bytes.NewReader(body))
	if err != nil {
		return StageResponse{}, err
	}
	request.Header.Set("Content-Type", "application/octet-stream")
	var response StageResponse
	if err := c.do(request, &response); err != nil {
		return StageResponse{}, err
	}
	return response, nil
}

func (c *Client) Complete(ctx context.Context, baseURL, attemptID string, req CompleteRequest) error {
	return c.postJSON(ctx, http.MethodPost, baseURL+"/v1/attempts/"+attemptID+"/complete", req, nil)
}

func (c *Client) postJSON(ctx context.Context, method, url string, body any, out any) error {
	payload, err := json.Marshal(body)
	if err != nil {
		return err
	}
	request, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(payload))
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", "application/json")
	return c.do(request, out)
}

func (c *Client) do(request *http.Request, out any) error {
	response, err := c.HTTP.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	data, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if err != nil {
		return err
	}
	if response.StatusCode >= 200 && response.StatusCode < 300 {
		if out == nil || len(data) == 0 {
			return nil
		}
		return json.Unmarshal(data, out)
	}
	var errorResponse ErrorResponse
	_ = json.Unmarshal(data, &errorResponse)
	if errorResponse.Error == "" {
		errorResponse.Error = http.StatusText(response.StatusCode)
	}
	return &HTTPError{StatusCode: response.StatusCode, Message: errorResponse.Error, Category: errorResponse.FailureCategory}
}

type HTTPError struct {
	StatusCode int
	Message    string
	Category   string
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("participant callback http %d: %s", e.StatusCode, e.Message)
}

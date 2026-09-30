package participant

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"opp297/topk/internal/domain"
)

type Client struct {
	BaseURL string
	HTTP    *http.Client
}

type MultiClient struct {
	Clients map[int]*Client
}

func NewMultiClient(urls map[int]string) *MultiClient {
	c := &MultiClient{Clients: make(map[int]*Client, len(urls))}
	for id, url := range urls {
		c.Clients[id] = &Client{BaseURL: url, HTTP: http.DefaultClient}
	}
	return c
}

func (c *MultiClient) ReadPage(ctx context.Context, req domain.PageRequest) (domain.PageResponse, error) {
	client, ok := c.Clients[req.ShardID]
	if !ok {
		return domain.PageResponse{}, fmt.Errorf("%w: no participant for shard %d", domain.ErrUnavailable, req.ShardID)
	}
	return client.ReadPage(ctx, req)
}

func (c *Client) ReadPage(ctx context.Context, req domain.PageRequest) (domain.PageResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return domain.PageResponse{}, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL, bytes.NewReader(body))
	if err != nil {
		return domain.PageResponse{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")

	httpClient := c.HTTP
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	resp, err := httpClient.Do(httpReq)
	if err != nil {
		return domain.PageResponse{}, err
	}
	defer resp.Body.Close()

	var page domain.PageResponse
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&page); err != nil {
		return domain.PageResponse{}, fmt.Errorf("decode shard response: %w", err)
	}
	if resp.StatusCode == http.StatusGatewayTimeout {
		return page, context.DeadlineExceeded
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return page, fmt.Errorf("participant HTTP %d", resp.StatusCode)
	}
	return page, nil
}

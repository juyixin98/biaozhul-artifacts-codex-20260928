package coordinator

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"sync/atomic"

	"topk-demo/internal/safolog"
)

type requestIDKey struct{}

var requestOffset atomic.Int64

func requestIDMiddleware(log *slog.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			id := strings.TrimSpace(r.Header.Get("X-Request-ID"))
			if id == "" {
				id = fmt.Sprintf("req-%d", requestOffset.Add(1))
			}
			ctx := context.WithValue(r.Context(), requestIDKey{}, id)
			w.Header().Set("X-Request-ID", id)
			safolog.Info(log, "http request", "request_id", id, "method", r.Method, "path", r.URL.Path)
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

func requestIDFrom(ctx context.Context, fallback string) string {
	if id, ok := ctx.Value(requestIDKey{}).(string); ok && id != "" {
		return id
	}
	return fallback
}

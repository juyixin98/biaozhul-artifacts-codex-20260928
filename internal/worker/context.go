package worker

import "context"

type serverURLContextKey struct{}

func WithServerURL(ctx context.Context, url string) context.Context {
	return context.WithValue(ctx, serverURLContextKey{}, url)
}

func ServerURL(ctx context.Context) string {
	value, _ := ctx.Value(serverURLContextKey{}).(string)
	return value
}

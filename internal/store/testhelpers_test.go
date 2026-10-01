package store

import (
	"net/url"
	"os"
	"strings"
)

func testDatabaseURL() string {
	if v := os.Getenv("DATABASE_URL"); v != "" {
		return v
	}
	return "postgres://admin:specbatch@127.0.0.1:5432/specbatch?sslmode=disable"
}

func withSchema(raw, schema string) string {
	u, err := url.Parse(raw)
	if err != nil {
		return raw
	}
	q := u.Query()
	if existing := q.Get("search_path"); existing != "" {
		q.Set("search_path", existing+","+schema)
	} else {
		q.Set("search_path", schema)
	}
	u.RawQuery = q.Encode()
	out := u.String()
	return strings.ReplaceAll(out, "search_path="+schema, "search_path="+schema)
}

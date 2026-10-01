package config

import "testing"

func TestWorkerConfigRejectsEmptyCoordinator(t *testing.T) {
	_, err := WorkerFromFlags([]string{"-coordinator-url=", "-worker-id=", "-staging-root="})
	if err == nil {
		t.Fatal("expected validation error")
	}
}

func TestRedactedDSN(t *testing.T) {
	got := RedactedDSN("postgres://admin:secret@127.0.0.1:5432/specbatch?sslmode=disable")
	if got == "" || contains(got, "secret") || !contains(got, "redacted") {
		t.Fatalf("password not redacted: %s", got)
	}
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (func() bool {
		for i := 0; i+len(sub) <= len(s); i++ {
			if s[i:i+len(sub)] == sub {
				return true
			}
		}
		return false
	}())
}

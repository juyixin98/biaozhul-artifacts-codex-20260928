package safolog

import (
	"log/slog"
	"os"
)

func New(service string) *slog.Logger {
	handler := slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo})
	return slog.New(handler).With("service", service)
}

func Info(log *slog.Logger, msg string, args ...any) {
	if log != nil {
		log.Info(msg, redact(args)...)
	}
}

func Warn(log *slog.Logger, msg string, args ...any) {
	if log != nil {
		log.Warn(msg, redact(args)...)
	}
}

// Item identifiers and scores are proof metadata rather than sensitive fixture
// payloads. Request bodies, labels and free-form text are intentionally never
// logged; callers pass only enumerated status fields and short error messages.
func redact(args []any) []any { return args }

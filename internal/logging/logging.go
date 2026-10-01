package logging

import (
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"
)

type Logger struct {
	mu  sync.Mutex
	w   io.Writer
	now func() time.Time
}

func New(path string, stdout bool) (*Logger, error) {
	writers := []io.Writer{}
	var f *os.File
	if path != "" {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return nil, err
		}
		var err error
		f, err = os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			return nil, err
		}
		writers = append(writers, f)
	}
	if stdout || path == "" {
		writers = append(writers, os.Stdout)
	}
	if f != nil && stdout {
		// Both paths receive the same JSON line.
	}
	return &Logger{w: io.MultiWriter(writers...), now: time.Now}, nil
}

func (l *Logger) Log(level, event string, fields map[string]any) {
	if l == nil {
		return
	}
	if fields == nil {
		fields = map[string]any{}
	}
	fields["ts"] = l.now().UTC().Format(time.RFC3339Nano)
	fields["level"] = level
	fields["event"] = event
	b, _ := json.Marshal(fields)
	l.mu.Lock()
	defer l.mu.Unlock()
	_, _ = l.w.Write(append(b, '\n'))
}

func (l *Logger) Info(event string, fields map[string]any)  { l.Log("info", event, fields) }
func (l *Logger) Error(event string, fields map[string]any) { l.Log("error", event, fields) }
func (l *Logger) Debug(event string, fields map[string]any) { l.Log("debug", event, fields) }

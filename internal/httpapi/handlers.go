package httpapi

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strconv"

	"diststats/internal/apperror"
	"diststats/internal/protocol"
	"diststats/internal/service"
)

const maxRequestBody = 4 << 20

type runHandler func(http.ResponseWriter, *http.Request, *service.RunHandle, json.RawMessage)

func (s *Server) withRun(jobPath string, h runHandler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		body, decodeBody, err := readBody(r)
		if err != nil {
			writeError(w, err)
			return
		}
		var jobID *string
		if jobPath != "" {
			id := r.PathValue(jobPath)
			jobID = &id
		}
		var request any
		if len(body) > 0 {
			if err := json.Unmarshal(body, &request); err != nil {
				request = map[string]string{"invalid_body": string(body)}
			}
		}
		run, err := s.service.StartRun(r.Context(), r.Method, r.URL.Path, request, jobID)
		if err != nil {
			writeError(w, err)
			return
		}
		setRunHeaders(w, run)
		h(w, r, run, decodeBody)
	}
}

func readBody(r *http.Request) (json.RawMessage, json.RawMessage, error) {
	if r.Body == nil {
		return nil, nil, nil
	}
	r.Body = http.MaxBytesReader(nil, r.Body, maxRequestBody)
	b, err := io.ReadAll(r.Body)
	if err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			return nil, nil, apperror.ResourceExhausted("REQUEST_TOO_LARGE",
				"request body exceeds 4 MiB limit")
		}
		return nil, nil, apperror.Unavailable("REQUEST_READ_FAILED",
			"cannot read request body", err)
	}
	if len(b) == 0 {
		return nil, nil, nil
	}
	return b, b, nil
}

func (s *Server) finish(w http.ResponseWriter, r *http.Request, run *service.RunHandle, status int, err error, v any) {
	if completeErr := s.service.CompleteRun(r.Context(), run, status, err); completeErr != nil {
		s.log.Error("failed to complete run", "run_id", run.ID, "error", completeErr)
	}
	setRunHeaders(w, run)
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, status, v)
}

func setRunHeaders(w http.ResponseWriter, run *service.RunHandle) {
	if run == nil {
		return
	}
	w.Header().Set("X-Run-ID", run.ID)
	w.Header().Set("X-Run-Number", strconv.FormatInt(run.Number, 10))
}

func decodeRequest(raw json.RawMessage, dst any) error {
	if len(raw) == 0 {
		return apperror.InvalidInput("MISSING_BODY", "request body is required")
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		return apperror.InvalidInput("MALFORMED_JSON", "request body does not match protocol")
	}
	if dec.More() {
		return apperror.InvalidInput("MALFORMED_JSON", "request body must contain one JSON object")
	}
	return nil
}

func (s *Server) createJob(w http.ResponseWriter, r *http.Request) {
	s.withRun("", func(w http.ResponseWriter, r *http.Request, run *service.RunHandle, raw json.RawMessage) {
		var req protocol.CreateJobRequest
		if err := decodeRequest(raw, &req); err != nil {
			s.finish(w, r, run, http.StatusBadRequest, err, nil)
			return
		}
		job, err := s.service.CreateJob(r.Context(), run, req)
		if err != nil {
			s.finish(w, r, run, statusForError(err), err, nil)
			return
		}
		resp, err := s.service.GetJobResponse(r.Context(), job.ID)
		s.finish(w, r, run, http.StatusCreated, err, resp)
	})(w, r)
}

func (s *Server) getJob(w http.ResponseWriter, r *http.Request) {
	s.withRun("id", func(w http.ResponseWriter, r *http.Request, run *service.RunHandle, _ json.RawMessage) {
		resp, err := s.service.GetJobResponse(r.Context(), r.PathValue("id"))
		s.finish(w, r, run, http.StatusOK, err, resp)
	})(w, r)
}

func (s *Server) submitShard(w http.ResponseWriter, r *http.Request) {
	s.withRun("id", func(w http.ResponseWriter, r *http.Request, run *service.RunHandle, raw json.RawMessage) {
		var req protocol.SubmitShardRequest
		if err := decodeRequest(raw, &req); err != nil {
			s.finish(w, r, run, http.StatusBadRequest, err, nil)
			return
		}
		resp, err := s.service.SubmitShard(r.Context(), run, r.PathValue("id"), req)
		s.finish(w, r, run, http.StatusAccepted, err, resp)
	})(w, r)
}

func (s *Server) finalizeJob(w http.ResponseWriter, r *http.Request) {
	s.withRun("id", func(w http.ResponseWriter, r *http.Request, run *service.RunHandle, _ json.RawMessage) {
		_, err := s.service.Finalize(r.Context(), run, r.PathValue("id"))
		if err != nil {
			s.finish(w, r, run, statusForError(err), err, nil)
			return
		}
		resp, err := s.service.GetJobResponse(r.Context(), r.PathValue("id"))
		s.finish(w, r, run, http.StatusOK, err, resp)
	})(w, r)
}

func (s *Server) getResult(w http.ResponseWriter, r *http.Request) {
	s.withRun("id", func(w http.ResponseWriter, r *http.Request, run *service.RunHandle, _ json.RawMessage) {
		resp, err := s.service.GetResult(r.Context(), r.PathValue("id"))
		s.finish(w, r, run, http.StatusOK, err, resp)
	})(w, r)
}

func (s *Server) replay(w http.ResponseWriter, r *http.Request) {
	s.withRun("id", func(w http.ResponseWriter, r *http.Request, run *service.RunHandle, _ json.RawMessage) {
		order := r.URL.Query().Get("order")
		if order == "" {
			order = "sequential"
		}
		compare := r.URL.Query().Get("compare") != "false"
		resp, err := s.service.Replay(r.Context(), run, r.PathValue("id"), order, compare)
		s.finish(w, r, run, http.StatusOK, err, resp)
	})(w, r)
}

func (s *Server) getRun(w http.ResponseWriter, r *http.Request) {
	runMW := s.withRun("", func(w http.ResponseWriter, r *http.Request, outer *service.RunHandle, _ json.RawMessage) {
		number, err := strconv.ParseInt(r.PathValue("number"), 10, 64)
		if err != nil {
			s.finish(w, r, outer, http.StatusBadRequest,
				apperror.InvalidInput("INVALID_RUN_NUMBER", "run number must be a positive integer"), nil)
			return
		}
		resp, err := s.service.GetRun(r.Context(), number)
		s.finish(w, r, outer, http.StatusOK, err, resp)
	})
	runMW(w, r)
}

func (s *Server) listJobRuns(w http.ResponseWriter, r *http.Request) {
	s.withRun("id", func(w http.ResponseWriter, r *http.Request, run *service.RunHandle, _ json.RawMessage) {
		limit := 100
		if v := r.URL.Query().Get("limit"); v != "" {
			n, err := strconv.Atoi(v)
			if err != nil || n <= 0 {
				s.finish(w, r, run, http.StatusBadRequest,
					apperror.InvalidInput("INVALID_LIMIT", "limit must be a positive integer"), nil)
				return
			}
			limit = n
		}
		resp, err := s.service.ListJobRuns(r.Context(), r.PathValue("id"), run.ID, limit)
		s.finish(w, r, run, http.StatusOK, err, resp)
	})(w, r)
}

func statusForError(err error) int {
	ae := &apperror.Error{}
	if errors.As(err, &ae) {
		return statusForKind(ae.Kind)
	}
	return http.StatusServiceUnavailable
}

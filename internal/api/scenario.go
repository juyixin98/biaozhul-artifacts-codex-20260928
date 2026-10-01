package api

import (
	"encoding/json"
	"errors"
	"net/http"

	"checkpointlab/internal/kernel"
	"checkpointlab/internal/lab"
	"checkpointlab/internal/protocol"
	"checkpointlab/internal/source"
)

type runScenarioRequest struct {
	JobID    string          `json:"job_id"`
	Scenario source.Scenario `json:"scenario"`
	Resume   bool            `json:"resume"`
}

func (a *App) runScenario(w http.ResponseWriter, r *http.Request) {
	var req runScenarioRequest
	if !decode(w, r, &req) {
		return
	}
	if req.JobID == "" {
		req.JobID = req.Scenario.Name
	}
	runner := lab.NewRunner(a.store)
	var report lab.Report
	var op *kernel.Operator
	var err error
	if req.Resume {
		op, report, err = runner.Resume(r.Context(), req.JobID, req.Scenario, requestID(r))
	} else {
		op, report, err = runner.Run(r.Context(), req.JobID, req.Scenario, requestID(r))
	}
	if op != nil {
		a.mu.Lock()
		a.jobs[req.JobID] = &job{
			mode: req.Scenario.Mode,
			cfg: kernel.Config{
				JobID:         req.JobID,
				Mode:          req.Scenario.Mode,
				GateCapacity:  req.Scenario.GateCapacity,
				InflightLimit: req.Scenario.InflightLimit,
			},
			op: op,
		}
		a.mu.Unlock()
	}
	if err != nil {
		writeError(w, r, http.StatusBadRequest, "SCENARIO_REJECTED", err.Error(), false)
		return
	}
	status := http.StatusOK
	if len(report.Failures) > 0 {
		status = http.StatusConflict
	}
	writeJSON(w, status, Envelope{
		OK:          len(report.Failures) == 0,
		RequestID:   requestID(r),
		Data:        report,
		Failure:     firstFailure(report.Failures),
		Uncertainty: report.Uncertainties,
	})
}

func decode(w http.ResponseWriter, r *http.Request, dst any) bool {
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		writeError(w, r, http.StatusBadRequest, "VALIDATION_REJECTED", err.Error(), false)
		return false
	}
	return true
}

func firstFailure(failures []lab.Failure) *FailureBody {
	if len(failures) == 0 {
		return nil
	}
	return &FailureBody{Category: failures[0].Category, Message: failures[0].Message}
}

func errorCategory(err error) string {
	var typed *protocol.TypedError
	if errors.As(err, &typed) {
		return string(typed.Category)
	}
	return "KERNEL_REJECTED"
}

func isRetriable(err error) bool {
	var typed *protocol.TypedError
	return errors.As(err, &typed) && typed.Retriable
}

func writeError(w http.ResponseWriter, r *http.Request, status int, category, message string, retriable bool) {
	writeJSON(w, status, Envelope{
		OK:        false,
		RequestID: requestID(r),
		Failure:   &FailureBody{Category: category, Message: message, Retriable: retriable},
	})
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

package postgres

import (
	"context"
	"encoding/json"

	"diststats/internal/protocol"
)

func (s *Store) RecordRejection(ctx context.Context, jobID, shardID, eventType string, class protocol.ErrorClass, code, detail string, shard *protocol.ShardRequest) (protocol.Event, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return protocol.Event{}, mapWriteError(err)
	}
	defer tx.Rollback(ctx)

	var runID, estimatorText string
	var received int
	var beforeJSON []byte
	err = tx.QueryRow(ctx, `
SELECT run_id, estimator, received_shards, current_summary
FROM diststats_b.jobs WHERE id = $1 FOR UPDATE`, jobID).Scan(&runID, &estimatorText, &received, &beforeJSON)
	if err != nil {
		return protocol.Event{}, mapReadError(err)
	}

	var sequence int
	err = tx.QueryRow(ctx, `SELECT COALESCE(MAX(sequence), 0) + 1 FROM diststats_b.events WHERE job_id = $1`, jobID).Scan(&sequence)
	if err != nil {
		return protocol.Event{}, mapReadError(err)
	}

	before, err := decodeSummary(beforeJSON)
	if err != nil {
		return protocol.Event{}, err
	}
	estimator := protocol.Estimator(estimatorText)
	view := before.View(estimator)
	var requestJSON []byte
	if shard != nil {
		requestJSON = mustJSON(*shard)
	}

	var event protocol.Event
	err = tx.QueryRow(ctx, `
INSERT INTO diststats_b.events (job_id, sequence, run_id, shard_id, type, reason, error_class, detail, request_json, before_state, after_state)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $10)
RETURNING created_at`,
		jobID, sequence, runID, shardID, eventType, code, string(class), detail, requestJSON, mustJSON(view)).Scan(&event.CreatedAt)
	if err != nil {
		return protocol.Event{}, mapWriteError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return protocol.Event{}, mapWriteError(err)
	}

	state := view
	event.Sequence = int64(sequence)
	event.JobID = jobID
	event.RunID = runID
	event.ShardID = shardID
	event.Type = eventType
	event.Reason = code
	event.ErrorClass = string(class)
	event.Detail = detail
	event.Before = &state
	event.After = &state
	return event, nil
}

func (s *Store) ListEvents(ctx context.Context, jobID string) ([]protocol.Event, error) {
	rows, err := s.pool.Query(ctx, `
SELECT e.sequence, e.job_id, e.run_id, e.shard_id, e.type, e.reason, e.error_class,
       e.detail, e.before_state, e.after_state, e.payload, e.request_json, e.created_at
FROM diststats_b.events e
WHERE e.job_id = $1
ORDER BY e.sequence ASC`, jobID)
	if err != nil {
		return nil, mapReadError(err)
	}
	defer rows.Close()

	var events []protocol.Event
	for rows.Next() {
		var event protocol.Event
		var beforeJSON, afterJSON, payloadJSON, requestJSON []byte
		if err := rows.Scan(&event.Sequence, &event.JobID, &event.RunID, &event.ShardID, &event.Type,
			&event.Reason, &event.ErrorClass, &event.Detail, &beforeJSON, &afterJSON, &payloadJSON,
			&requestJSON, &event.CreatedAt); err != nil {
			return nil, mapReadError(err)
		}
		event.Before, err = decodeView(beforeJSON)
		if err != nil {
			return nil, err
		}
		event.After, err = decodeView(afterJSON)
		if err != nil {
			return nil, err
		}
		if len(payloadJSON) > 0 {
			var shard protocol.ShardRequest
			if err := json.Unmarshal(payloadJSON, &shard); err != nil {
				return nil, protocol.NewServiceError(protocol.ClassStorage, "PAYLOAD_DECODE_FAILED",
					"corrupt shard payload", err)
			}
			event.Shard = &shard
		}
		if len(requestJSON) > 0 {
			var shard protocol.ShardRequest
			if err := json.Unmarshal(requestJSON, &shard); err != nil {
				return nil, protocol.NewServiceError(protocol.ClassStorage, "PAYLOAD_DECODE_FAILED",
					"corrupt rejected request payload", err)
			}
			event.Shard = &shard
		}
		events = append(events, event)
	}
	if err := rows.Err(); err != nil {
		return nil, mapReadError(err)
	}
	return events, nil
}

func decodeView(data []byte) (*protocol.SummaryView, error) {
	if len(data) == 0 {
		return nil, nil
	}
	var view protocol.SummaryView
	if err := json.Unmarshal(data, &view); err != nil {
		return nil, protocol.NewServiceError(protocol.ClassStorage, "STATE_DECODE_FAILED",
			"corrupt persisted state view", err)
	}
	return &view, nil
}

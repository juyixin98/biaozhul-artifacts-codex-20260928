package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"sort"

	"specbatch/internal/kernel"
	"specbatch/internal/model"
)

type JobStatusResponse struct {
	Job             model.Job       `json:"job"`
	ShardStatuses   []ShardStatus   `json:"shards"`
	Attempts        []model.Attempt `json:"attempts,omitempty"`
	WinnerByShard   map[string]int  `json:"winner_by_shard"`
	MergedRecordIDs []string        `json:"merged_record_ids,omitempty"`
}

type ShardStatus struct {
	ID     string `json:"id"`
	Index  int    `json:"index"`
	Status string `json:"status"`
	Winner int    `json:"winner_attempt_no,omitempty"`
}

func (s *Store) JobStatus(ctx context.Context, jobID string, includeAttempts bool) (JobStatusResponse, error) {
	job, err := s.GetJob(ctx, jobID)
	if err != nil {
		return JobStatusResponse{}, err
	}
	rows, err := s.db.QueryContext(ctx, `
SELECT sh.id,sh.idx,sh.status,COALESCE(a.attempt_no,0)
FROM shards sh
LEFT JOIN artifacts a ON a.job_id=sh.job_id AND a.shard_id=sh.id
WHERE sh.job_id=$1
ORDER BY sh.idx,sh.id`, jobID)
	if err != nil {
		return JobStatusResponse{}, err
	}
	defer rows.Close()
	var shards []ShardStatus
	winners := map[string]int{}
	for rows.Next() {
		var st ShardStatus
		if err := rows.Scan(&st.ID, &st.Index, &st.Status, &st.Winner); err != nil {
			return JobStatusResponse{}, err
		}
		shards = append(shards, st)
		if st.Winner > 0 {
			winners[st.ID] = st.Winner
		}
	}
	if err := rows.Err(); err != nil {
		return JobStatusResponse{}, err
	}
	resp := JobStatusResponse{Job: job, ShardStatuses: shards, WinnerByShard: winners}
	if includeAttempts {
		attempts, err := s.Attempts(ctx, jobID)
		if err != nil {
			return resp, err
		}
		resp.Attempts = attempts
	}
	if job.Status == model.JobSucceeded {
		merge, err := s.Merged(ctx, jobID)
		if err != nil {
			return resp, err
		}
		for _, rec := range merge.Records {
			resp.MergedRecordIDs = append(resp.MergedRecordIDs, rec.ID)
		}
	}
	return resp, nil
}

func (s *Store) Attempts(ctx context.Context, jobID string) ([]model.Attempt, error) {
	rows, err := s.db.QueryContext(ctx, `
SELECT job_id,shard_id,attempt_no,token,worker_id,kind,status,COALESCE(failure_category,''),progress,
       started_at,last_heartbeat_at,ended_at
FROM attempts WHERE job_id=$1 ORDER BY shard_id,attempt_no`, jobID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []model.Attempt
	for rows.Next() {
		var a model.Attempt
		var ended sql.NullTime
		if err := rows.Scan(&a.JobID, &a.ShardID, &a.AttemptNo, &a.Token, &a.WorkerID, &a.Kind,
			&a.Status, &a.FailureCategory, &a.Progress, &a.StartedAt, &a.LastHeartbeatAt, &ended); err != nil {
			return nil, err
		}
		if ended.Valid {
			t := ended.Time
			a.EndedAt = &t
		}
		out = append(out, a)
	}
	return out, rows.Err()
}

func (s *Store) ArtifactContent(ctx context.Context, jobID, shardID string) ([]byte, error) {
	var content []byte
	err := s.db.QueryRowContext(ctx, `SELECT content FROM artifacts WHERE job_id=$1 AND shard_id=$2`, jobID, shardID).Scan(&content)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, ErrNotFound
	}
	return content, err
}

func (s *Store) ArtifactRows(ctx context.Context, jobID string) (map[string][]model.OutputRecord, error) {
	rows, err := s.db.QueryContext(ctx, `
SELECT shard_id, content FROM artifacts WHERE job_id=$1 ORDER BY shard_id`, jobID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[string][]model.OutputRecord{}
	for rows.Next() {
		var shardID string
		var raw []byte
		if err := rows.Scan(&shardID, &raw); err != nil {
			return nil, err
		}
		var doc model.ArtifactDocument
		if err := json.Unmarshal(raw, &doc); err != nil {
			return nil, err
		}
		out[shardID] = doc.Records
	}
	return out, rows.Err()
}

func (s *Store) Merged(ctx context.Context, jobID string) (model.MergeDocument, error) {
	job, err := s.GetJob(ctx, jobID)
	if err != nil {
		return model.MergeDocument{}, err
	}
	if job.Status != model.JobSucceeded {
		// Caller may still inspect; avoid masking unknown/failure as success.
		return model.MergeDocument{}, errors.New("job is not succeeded")
	}
	rows, err := s.db.QueryContext(ctx, `
SELECT sh.idx,a.content
FROM artifacts a JOIN shards sh ON sh.job_id=a.job_id AND sh.id=a.shard_id
WHERE a.job_id=$1 ORDER BY sh.idx,a.shard_id`, jobID)
	if err != nil {
		return model.MergeDocument{}, err
	}
	defer rows.Close()
	var merged []model.OutputRecord
	seen := map[string]bool{}
	for rows.Next() {
		var idx int
		var raw []byte
		if err := rows.Scan(&idx, &raw); err != nil {
			return model.MergeDocument{}, err
		}
		var doc model.ArtifactDocument
		if err := json.Unmarshal(raw, &doc); err != nil {
			return model.MergeDocument{}, err
		}
		for _, rec := range doc.Records {
			if seen[rec.ID] {
				// A duplicated logical shard should never reach this path because
				// artifacts has one row per shard; keep an explicit guard anyway.
				return model.MergeDocument{}, errors.New("duplicate merged record id")
			}
			seen[rec.ID] = true
			merged = append(merged, rec)
		}
	}
	if err := rows.Err(); err != nil {
		return model.MergeDocument{}, err
	}
	sort.Slice(merged, func(i, j int) bool {
		if merged[i].ShardID != merged[j].ShardID {
			return merged[i].ShardID < merged[j].ShardID
		}
		return merged[i].ID < merged[j].ID
	})
	return model.MergeDocument{
		SchemaVersion: "specbatch/merge/v1", JobID: jobID,
		RecordCount: len(merged), Records: merged,
	}, nil
}

func (s *Store) Events(ctx context.Context, jobID string, limit int) ([]model.Event, error) {
	if limit <= 0 || limit > 1000 {
		limit = 200
	}
	args := []any{jobID, limit}
	q := `
SELECT seq,job_id,shard_id,attempt_no,worker_id,type,status,category,reason,payload,created_at
FROM events WHERE job_id=$1 OR $1=''
ORDER BY seq DESC LIMIT $2`
	rows, err := s.db.QueryContext(ctx, q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []model.Event
	for rows.Next() {
		var e model.Event
		if err := rows.Scan(&e.Seq, &e.JobID, &e.ShardID, &e.AttemptNo, &e.WorkerID, &e.Type,
			&e.Status, &e.Category, &e.Reason, &e.Payload, &e.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, e)
	}
	return out, rows.Err()
}

func kernelVersion() string { return kernel.Version }

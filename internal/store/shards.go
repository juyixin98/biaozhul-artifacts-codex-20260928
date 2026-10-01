package store

import (
	"context"
)

func scanShard(row interface{ Scan(...any) error }) (Shard, error) {
	var s Shard
	err := row.Scan(&s.JobID, &s.ShardID, &s.Seq, &s.N, &s.Weight,
		&s.Summary, &s.SubmittedRunID, &s.CreatedAt)
	if err != nil {
		return Shard{}, mapDBError(err)
	}
	return s, nil
}

func GetShard(ctx context.Context, q queryer, jobID, shardID string) (Shard, error) {
	row := q.QueryRowContext(ctx, `
SELECT job_id, shard_id, seq, n, weight, summary, submitted_run_id, created_at
FROM shards WHERE job_id = $1 AND shard_id = $2`, jobID, shardID)
	return scanShard(row)
}

func InsertShard(ctx context.Context, q queryer, shard Shard) error {
	_, err := q.ExecContext(ctx, `
INSERT INTO shards
    (job_id, shard_id, seq, n, weight, summary, submitted_run_id)
VALUES ($1, $2, $3, $4, $5, $6, $7)`,
		shard.JobID, shard.ShardID, shard.Seq, shard.N, shard.Weight,
		[]byte(shard.Summary), shard.SubmittedRunID)
	if err != nil {
		return mapDBError(err)
	}
	return nil
}

func (d *DB) NextShardSeq(ctx context.Context, jobID string) (int64, error) {
	return NextShardSeq(ctx, d.db, jobID)
}

func NextShardSeq(ctx context.Context, q queryer, jobID string) (int64, error) {
	var seq int64
	err := q.QueryRowContext(ctx, `
SELECT COALESCE(MAX(seq) + 1, 1) FROM shards WHERE job_id = $1`, jobID).
		Scan(&seq)
	if err != nil {
		return 0, mapDBError(err)
	}
	return seq, nil
}

func (d *DB) ListShards(ctx context.Context, jobID string) ([]Shard, error) {
	rows, err := d.db.QueryContext(ctx, `
SELECT job_id, shard_id, seq, n, weight, summary, submitted_run_id, created_at
FROM shards WHERE job_id = $1 ORDER BY seq`, jobID)
	if err != nil {
		return nil, mapDBError(err)
	}
	defer rows.Close()

	var out []Shard
	for rows.Next() {
		s, err := scanShard(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	if err := rows.Err(); err != nil {
		return nil, mapDBError(err)
	}
	return out, nil
}

func (d *DB) ShardExists(ctx context.Context, jobID, shardID string) (bool, error) {
	var exists bool
	err := d.db.QueryRowContext(ctx,
		`SELECT EXISTS(SELECT 1 FROM shards WHERE job_id=$1 AND shard_id=$2)`,
		jobID, shardID).Scan(&exists)
	if err != nil {
		return false, mapDBError(err)
	}
	return exists, nil
}

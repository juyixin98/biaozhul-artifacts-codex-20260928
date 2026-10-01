package store

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"

	"opp300-speculative-batch/internal/idgen"
	"opp300-speculative-batch/internal/protocol"
)

type Store struct {
	pool *pgxpool.Pool
}

func New(ctx context.Context, databaseURL string) (*Store, func(), error) {
	config, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		return nil, nil, err
	}
	config.MaxConns = 8
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		return nil, nil, err
	}
	return &Store{pool: pool}, pool.Close, nil
}

func (s *Store) Ping(ctx context.Context) error {
	return s.pool.Ping(ctx)
}

type CreateJobResult struct {
	Job           protocol.Job
	Shards        []protocol.Shard
	AlreadyExists bool
}

func (s *Store) CreateJob(ctx context.Context, spec protocol.JobSpec) (CreateJobResult, error) {
	if len(spec.Shards) == 0 {
		return CreateJobResult{}, errors.Join(ErrInvalidState, errors.New("job requires at least one shard"))
	}
	if len(spec.Shards) > 1024 {
		return CreateJobResult{}, errors.Join(ErrInvalidState, errors.New("job exceeds 1024 shards"))
	}
	policy := spec.Policy.Normalized()
	jobID, err := idgen.New("job")
	if err != nil {
		return CreateJobResult{}, err
	}

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return CreateJobResult{}, err
	}
	defer tx.Rollback(ctx)

	_, err = tx.Exec(ctx, `
INSERT INTO jobs (id, idempotency_key, state, version, total_shards,
  max_attempts_per_shard, max_active_attempts, lease_ms, speculate_after_ms, heartbeat_interval_ms)
VALUES ($1,$2,'pending',0,$3,$4,$5,$6,$7,$8)`,
		jobID, spec.IDempotencyKey, len(spec.Shards),
		policy.MaxAttemptsPerShard, policy.MaxActiveAttempts, policy.LeaseMS,
		policy.SpeculateAfterMS, policy.HeartbeatIntervalMS)
	if err != nil {
		var pgErr *pgconn.PgError
		if errors.As(err, &pgErr) && pgErr.Code == "23505" {
			return s.loadJobByIdempotencyKey(ctx, spec.IDempotencyKey)
		}
		return CreateJobResult{}, err
	}

	shards := make([]protocol.Shard, 0, len(spec.Shards))
	seenOrdinals := map[int]bool{}
	seenKeys := map[string]bool{}
	for _, in := range spec.Shards {
		if in.Key == "" || in.Input == "" || len(in.Key) > 256 || len(in.Input) > (4<<20) {
			return CreateJobResult{}, errors.Join(ErrInvalidState, errors.New("shard key and input are required"))
		}
		if in.Ordinal < 0 || seenOrdinals[in.Ordinal] {
			return CreateJobResult{}, errors.Join(ErrInvalidState, errors.New("duplicate or invalid ordinal"))
		}
		if seenKeys[in.Key] {
			return CreateJobResult{}, errors.Join(ErrInvalidState, errors.New("duplicate shard key"))
		}
		if in.Behavior.FailureCategory != "" && !validFailureCategory(in.Behavior.FailureCategory) {
			return CreateJobResult{}, errors.Join(ErrInvalidState, errors.New("invalid synthetic failure category"))
		}
		seenOrdinals[in.Ordinal] = true
		seenKeys[in.Key] = true

		shardID, err := idgen.New("shd")
		if err != nil {
			return CreateJobResult{}, err
		}
		behavior, err := json.Marshal(in.Behavior)
		if err != nil {
			return CreateJobResult{}, err
		}
		_, err = tx.Exec(ctx, `
INSERT INTO shards (id, job_id, ordinal, shard_key, input, behavior_json, state)
VALUES ($1,$2,$3,$4,$5,$6,'pending')`,
			shardID, jobID, in.Ordinal, in.Key, in.Input, behavior)
		if err != nil {
			return CreateJobResult{}, err
		}
		shards = append(shards, protocol.Shard{
			ID: shardID, JobID: jobID, Ordinal: in.Ordinal, Key: in.Key,
			Input: in.Input, Behavior: in.Behavior, State: protocol.ShardPending,
		})
	}

	if err := insertEvent(ctx, tx, jobID, "", "", "JOB_SUBMITTED", 0, "", map[string]any{
		"input_id":     spec.IDempotencyKey,
		"total_shards": len(spec.Shards),
		"policy":       policy,
	}); err != nil {
		return CreateJobResult{}, err
	}
	job, err := scanJob(tx.QueryRow(ctx, `SELECT `+jobColumns+` FROM jobs WHERE id=$1`, jobID))
	if err != nil {
		return CreateJobResult{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return CreateJobResult{}, err
	}
	return CreateJobResult{Job: job, Shards: shards}, nil
}

func (s *Store) loadJobByIdempotencyKey(ctx context.Context, key string) (CreateJobResult, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return CreateJobResult{}, err
	}
	defer tx.Rollback(ctx)

	job, err := scanJob(tx.QueryRow(ctx, `SELECT `+jobColumns+` FROM jobs WHERE idempotency_key=$1 FOR UPDATE`, key))
	if err != nil {
		return CreateJobResult{}, err
	}
	shards, err := queryShards(ctx, tx, job.ID)
	if err != nil {
		return CreateJobResult{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return CreateJobResult{}, err
	}
	return CreateJobResult{Job: job, Shards: shards, AlreadyExists: true}, nil
}

func mapNotFound(err error) error {
	if errors.Is(err, pgx.ErrNoRows) {
		return ErrNotFound
	}
	return err
}

package shardstore

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"topk-demo/internal/protocol"
)

var ErrBadGeneration = errors.New("unknown or stale data generation")

type PostgresStore struct {
	pool *pgxpool.Pool
}

func NewPostgresStore(ctx context.Context, dsn string) (*PostgresStore, error) {
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return nil, fmt.Errorf("connect postgres: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("ping postgres: %w", err)
	}
	return &PostgresStore{pool: pool}, nil
}

func (s *PostgresStore) Close() { s.pool.Close() }

func (s *PostgresStore) checkGeneration(ctx context.Context, generation string) error {
	var active string
	err := s.pool.QueryRow(ctx, `SELECT generation FROM generations WHERE generation = $1`, generation).Scan(&active)
	if errors.Is(err, pgx.ErrNoRows) {
		return ErrBadGeneration
	}
	if err != nil {
		return err
	}
	return nil
}

func (s *PostgresStore) Bound(ctx context.Context, generation, shardID, rawCursor string, cursor Cursor) (*protocol.BoundResponse, error) {
	if err := s.checkGeneration(ctx, generation); err != nil {
		return nil, err
	}
	resp := &protocol.BoundResponse{
		Generation:  generation,
		ShardID:     shardID,
		Cursor:      rawCursor,
		BoundProbes: 1,
	}
	var item protocol.Candidate
	err := s.pool.QueryRow(ctx, boundSQL, generation, shardID, cursor.ResumeScore, cursor.ResumeItemID, cursor.Include).
		Scan(&item.ItemID, &item.Score)
	if errors.Is(err, pgx.ErrNoRows) {
		resp.Exhausted = true
		return resp, nil
	}
	if err != nil {
		return nil, err
	}
	resp.Upper = &item
	resp.ReadRows = 1
	return resp, nil
}

func (s *PostgresStore) Scan(ctx context.Context, generation, shardID, rawCursor string, cursor Cursor, pageSize int) (*protocol.ScanResponse, error) {
	if pageSize <= 0 {
		return nil, fmt.Errorf("page_size must be positive")
	}
	if err := s.checkGeneration(ctx, generation); err != nil {
		return nil, err
	}
	rows, err := s.pool.Query(ctx, scanSQL, generation, shardID, cursor.ResumeScore, cursor.ResumeItemID, cursor.Include, pageSize+1)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	items := make([]protocol.Candidate, 0, pageSize+1)
	for rows.Next() {
		var item protocol.Candidate
		if err := rows.Scan(&item.ItemID, &item.Score); err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	resp := &protocol.ScanResponse{
		Generation:         generation,
		ShardID:            shardID,
		Cursor:             rawCursor,
		Items:              []protocol.Candidate{},
		ItemsIncludedUpper: cursor.Include,
		ReturnedRows:       0,
	}
	if cursor.Include {
		requested := protocol.Candidate{ItemID: cursor.ResumeItemID, Score: cursor.ResumeScore}
		resp.RequestedUpper = &requested
	}
	if len(items) == 0 {
		resp.Exhausted = true
		return resp, nil
	}
	if len(items) <= pageSize {
		resp.Items = items
		resp.ReturnedRows = int64(len(items))
		resp.ReadRows = int64(len(items))
		resp.Exhausted = true
		return resp, nil
	}
	upper := items[pageSize]
	resp.Items = items[:pageSize]
	resp.ReturnedRows = int64(pageSize)
	resp.ReadRows = int64(pageSize + 1)
	resp.BoundProbes = 1
	resp.NextUpper = &upper
	resp.NextCursorAt = &upper
	resp.NextCursor = EncodeCursor(cursorFor(upper.Score, upper.ItemID))
	return resp, nil
}

const boundSQL = `
SELECT item_id, score
FROM ranked_items
WHERE generation = $1
  AND shard_id = $2
  AND ($3 = 0 AND $4 = '' OR ($5 AND score = $3 AND item_id = $4) OR score < $3 OR score = $3 AND item_id > $4)
ORDER BY score DESC, item_id ASC
LIMIT 1`

const scanSQL = `
SELECT item_id, score
FROM ranked_items
WHERE generation = $1
  AND shard_id = $2
  AND ($3 = 0 AND $4 = '' OR ($5 AND score = $3 AND item_id = $4) OR score < $3 OR score = $3 AND item_id > $4)
ORDER BY score DESC, item_id ASC
LIMIT $5`

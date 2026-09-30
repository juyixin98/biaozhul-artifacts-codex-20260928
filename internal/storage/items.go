package storage

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"sort"

	"github.com/jackc/pgx/v5"

	"opp297/topk/internal/domain"
)

type ItemRecord struct {
	Generation domain.GenerationID
	ShardID    int
	ItemID     domain.ItemID
	Score      int64
	Label      string
}

type PageRow struct {
	Item  domain.Candidate
	Label string
}

func (s *Store) GenerationExists(ctx context.Context, gen domain.GenerationID) (bool, error) {
	var exists bool
	err := s.pool.QueryRow(ctx,
		`SELECT EXISTS(SELECT 1 FROM generations WHERE generation_id=$1)`, string(gen)).Scan(&exists)
	return exists, err
}

func (s *Store) ReadItemPage(ctx context.Context, gen domain.GenerationID, shardID, limit int, cursor *domain.Key) ([]PageRow, error) {
	exists, err := s.GenerationExists(ctx, gen)
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, fmt.Errorf("%w: generation %s", domain.ErrUnavailable, gen)
	}
	var cursorScore *int64
	var cursorID *string
	if cursor != nil {
		cursorScore = &cursor.Score
		id := string(cursor.ID)
		cursorID = &id
	}
	rows, err := s.pool.Query(ctx, `
SELECT item_id, score, sensitive_label
FROM items
WHERE generation_id=$1
  AND shard_id=$2
  AND ($3::bigint IS NULL OR score < $3 OR (score = $3 AND item_id > $4::text))
ORDER BY score DESC, item_id ASC
LIMIT $5`,
		string(gen), shardID, cursorScore, cursorID, limit+1)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []PageRow
	for rows.Next() {
		var row PageRow
		var id string
		if err := rows.Scan(&id, &row.Item.Score, &row.Label); err != nil {
			return nil, err
		}
		row.Item.ID = domain.ItemID(id)
		out = append(out, row)
	}
	return out, rows.Err()
}

func Checksum(records []ItemRecord) string {
	sort.Slice(records, func(i, j int) bool {
		a, b := records[i], records[j]
		if a.ShardID != b.ShardID {
			return a.ShardID < b.ShardID
		}
		if a.Score != b.Score {
			return a.Score > b.Score
		}
		return a.ItemID < b.ItemID
	})
	h := sha256.New()
	for _, r := range records {
		fmt.Fprintf(h, "%d\x00%d\x00%s\x00%s\x00", r.ShardID, r.Score, r.ItemID, r.Label)
	}
	return hex.EncodeToString(h.Sum(nil))
}

func (s *Store) SeedGeneration(ctx context.Context, gen domain.GenerationID, records []ItemRecord) error {
	checksum := Checksum(records)
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	var existingCount int
	var existingChecksum string
	err = tx.QueryRow(ctx,
		`SELECT item_count, checksum FROM generations WHERE generation_id=$1`, string(gen)).
		Scan(&existingCount, &existingChecksum)
	switch {
	case errors.Is(err, pgx.ErrNoRows):
		if _, err := tx.Exec(ctx,
			`INSERT INTO generations(generation_id,item_count,checksum) VALUES($1,$2,$3)`,
			string(gen), len(records), checksum); err != nil {
			return err
		}
	case err != nil:
		return err
	default:
		if existingCount != len(records) || existingChecksum != checksum {
			return fmt.Errorf("generation %s already exists with a different immutable world", gen)
		}
		return tx.Commit(ctx)
	}

	batch := &pgx.Batch{}
	for _, r := range records {
		if r.Generation != gen {
			return fmt.Errorf("record generation %s does not match %s", r.Generation, gen)
		}
		batch.Queue(`INSERT INTO items(generation_id,shard_id,item_id,score,sensitive_label)
VALUES($1,$2,$3,$4,$5)
ON CONFLICT (generation_id,item_id) DO NOTHING`,
			string(r.Generation), r.ShardID, string(r.ItemID), r.Score, r.Label)
	}
	br := tx.SendBatch(ctx, batch)
	for range records {
		tag, err := br.Exec()
		if err != nil {
			_ = br.Close()
			return err
		}
		if tag.RowsAffected() != 1 {
			_ = br.Close()
			return errors.New("seed contains duplicate global item id")
		}
	}
	if err := br.Close(); err != nil {
		return err
	}
	return tx.Commit(ctx)
}

package seed

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5/pgxpool"

	"topk-demo/internal/fixture"
)

func Run(ctx context.Context, dsn string) error {
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return err
	}
	defer pool.Close()
	if err := pool.Ping(ctx); err != nil {
		return err
	}
	tx, err := pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	if _, err := tx.Exec(ctx, `
INSERT INTO generations (generation, description)
VALUES ($1, 'deterministic local fixture')
ON CONFLICT (generation) DO UPDATE SET description = EXCLUDED.description`, fixture.Generation); err != nil {
		return err
	}
	if _, err := tx.Exec(ctx, `DELETE FROM ranked_items WHERE generation = $1`, fixture.Generation); err != nil {
		return err
	}
	for _, item := range fixture.Items() {
		if _, err := tx.Exec(ctx, `
INSERT INTO ranked_items (generation, shard_id, item_id, score, label)
VALUES ($1,$2,$3,$4,$5)`,
			item.Generation, item.ShardID, item.Candidate.ItemID, item.Candidate.Score, item.Label); err != nil {
			return fmt.Errorf("insert %s/%s: %w", item.ShardID, item.Candidate.ItemID, err)
		}
	}
	return tx.Commit(ctx)
}

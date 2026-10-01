package store

import (
	"context"
	_ "embed"

	"diststats/internal/apperror"
)

//go:embed schema.sql
var migration string

func (d *DB) Migrate(ctx context.Context) error {
	if _, err := d.db.ExecContext(ctx, migration); err != nil {
		return apperror.Unavailable("DATABASE_MIGRATION_FAILED",
			"cannot initialize PostgreSQL schema", err)
	}
	return nil
}

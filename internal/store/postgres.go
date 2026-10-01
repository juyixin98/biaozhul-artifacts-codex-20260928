package store

import (
	"context"
	"database/sql"
	"errors"

	"diststats/internal/apperror"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	_ "github.com/jackc/pgx/v5/stdlib"
)

type DB struct {
	db *sql.DB
}

type Tx interface {
	Commit() error
	Rollback() error
}

func Open(ctx context.Context, url string) (*DB, error) {
	db, err := sql.Open("pgx", url)
	if err != nil {
		return nil, apperror.Unavailable("DATABASE_OPEN_FAILED",
			"cannot open PostgreSQL connection", err)
	}
	db.SetMaxOpenConns(12)
	db.SetMaxIdleConns(6)
	if err := db.PingContext(ctx); err != nil {
		_ = db.Close()
		return nil, apperror.Unavailable("DATABASE_UNAVAILABLE",
			"cannot connect to PostgreSQL", err)
	}
	return &DB{db: db}, nil
}

func (d *DB) Close() error { return d.db.Close() }

func (d *DB) Exec(ctx context.Context, query string, args ...any) (sql.Result, error) {
	res, err := d.db.ExecContext(ctx, query, args...)
	if err != nil {
		return nil, mapDBError(err)
	}
	return res, nil
}

func (d *DB) InTx(ctx context.Context, fn func(*sql.Tx) error) error {
	tx, err := d.db.BeginTx(ctx, nil)
	if err != nil {
		return mapDBError(err)
	}
	if err := fn(tx); err != nil {
		_ = tx.Rollback()
		return err
	}
	if err := tx.Commit(); err != nil {
		return mapDBError(err)
	}
	return nil
}

func mapDBError(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, sql.ErrNoRows) || errors.Is(err, pgx.ErrNoRows) {
		return apperror.NotFound("NOT_FOUND", "record does not exist")
	}
	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) {
		switch pgErr.Code {
		case "23505":
			return apperror.Conflict("DUPLICATE_SHARD", "shard id has already been merged")
		case "22P02":
			return apperror.InvalidInput("INVALID_ID", "identifier must be a UUID")
		case "53000", "53100", "53200", "53300", "53400":
			return apperror.ResourceExhausted("DATABASE_RESOURCE_EXHAUSTED",
				"PostgreSQL reported insufficient resources")
		}
	}
	return apperror.Unavailable("DATABASE_FAILED", "database operation failed", err)
}

func Map(err error) error { return mapDBError(err) }

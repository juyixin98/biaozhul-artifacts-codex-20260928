package storage

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/jackc/pgx/v5"

	"opp297/topk/internal/domain"
)

type PersistedKey = domain.Key

type PersistedStatus struct {
	ID         int         `json:"id"`
	Exhausted  bool        `json:"exhausted"`
	Active     bool        `json:"active"`
	UpperBound *domain.Key `json:"upper_bound,omitempty"`
	Cursor     *domain.Key `json:"cursor,omitempty"`
	Pages      int         `json:"pages"`
	RowsRead   int         `json:"rows_read"`
}

type PersistedEvent struct {
	Kind         string       `json:"kind"`
	ShardID      int          `json:"shard_id"`
	Round        int          `json:"round"`
	Cursor       *domain.Key  `json:"cursor,omitempty"`
	UpperBound   *domain.Key  `json:"upper_bound,omitempty"`
	Items        []domain.Key `json:"items,omitempty"`
	Threshold    *domain.Key  `json:"threshold,omitempty"`
	Reason       string       `json:"reason"`
	FailureClass string       `json:"failure_class,omitempty"`
	RowsRead     int          `json:"rows_read"`
	At           time.Time    `json:"at"`
}

type PersistedQuery struct {
	RequestID      string              `json:"request_id"`
	Generation     domain.GenerationID `json:"generation"`
	K              int                 `json:"k"`
	Exact          bool                `json:"exact"`
	Complete       bool                `json:"complete"`
	FailureClass   string              `json:"failure_class,omitempty"`
	FailureMessage string              `json:"failure_message,omitempty"`
	Threshold      *domain.Key         `json:"threshold,omitempty"`
	Rounds         int                 `json:"rounds"`
	TotalRowsRead  int                 `json:"total_rows_read"`
	StartedAt      time.Time           `json:"started_at"`
	CompletedAt    time.Time           `json:"completed_at"`
	Keys           []domain.Key        `json:"keys"`
	Statuses       []PersistedStatus   `json:"shard_statuses"`
	Events         []PersistedEvent    `json:"events"`
}

type Replay struct {
	Query    PersistedQuery    `json:"query"`
	Keys     []domain.Key      `json:"keys"`
	Statuses []PersistedStatus `json:"shard_statuses"`
	Events   []PersistedEvent  `json:"events"`
}

type scanner interface {
	Scan(any) error
}

func (s *Store) SaveQuery(ctx context.Context, q PersistedQuery) error {
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	if _, err := tx.Exec(ctx, `
INSERT INTO topk_queries
(request_id,generation_id,k,exact,complete,failure_class,failure_message,threshold,rounds,total_rows_read,started_at,completed_at)
VALUES($1,$2,$3,$4,$5,NULLIF($6,''),$7,$8,$9,$10,$11,$12)`,
		q.RequestID, string(q.Generation), q.K, q.Exact, q.Complete, q.FailureClass, q.FailureMessage,
		jsonArg(q.Threshold), q.Rounds, q.TotalRowsRead, q.StartedAt, q.CompletedAt); err != nil {
		return err
	}
	for pos, key := range q.Keys {
		if _, err := tx.Exec(ctx,
			`INSERT INTO topk_query_results(request_id,position,item_id,score) VALUES($1,$2,$3,$4)`,
			q.RequestID, pos+1, string(key.ID), key.Score); err != nil {
			return err
		}
	}
	for _, st := range q.Statuses {
		if _, err := tx.Exec(ctx, `
INSERT INTO topk_query_shard_states
(request_id,shard_id,exhausted,active,upper_bound_key,cursor_key,pages,rows_read)
VALUES($1,$2,$3,$4,$5,$6,$7,$8)`,
			q.RequestID, st.ID, st.Exhausted, st.Active, jsonArg(st.UpperBound), jsonArg(st.Cursor),
			st.Pages, st.RowsRead); err != nil {
			return err
		}
	}
	for seq, ev := range q.Events {
		if _, err := tx.Exec(ctx, `
INSERT INTO topk_query_events
(request_id,seq,kind,shard_id,round,cursor_key,upper_bound_key,threshold_key,item_keys,reason,failure_class,rows_read,occurred_at)
VALUES($1,$2,$3,NULLIF($4,-1),$5,$6,$7,$8,$9,$10,NULLIF($11,''),$12,$13)`,
			q.RequestID, seq, ev.Kind, nullNegative(ev.ShardID), ev.Round,
			jsonArg(ev.Cursor), jsonArg(ev.UpperBound), jsonArg(ev.Threshold), jsonArg(ev.Items),
			ev.Reason, ev.FailureClass, ev.RowsRead, ev.At); err != nil {
			return err
		}
	}
	return tx.Commit(ctx)
}

func (s *Store) GetReplay(ctx context.Context, requestID string) (Replay, error) {
	var out Replay
	q := &out.Query
	var failureClass *string
	var failureMessage *string
	err := s.pool.QueryRow(ctx, `
SELECT request_id,generation_id,k,exact,complete,failure_class,failure_message,threshold,rounds,total_rows_read,started_at,completed_at
FROM topk_queries WHERE request_id=$1`, requestID).Scan(
		&q.RequestID, &q.Generation, &q.K, &q.Exact, &q.Complete, &failureClass, &failureMessage,
		jsonScan(&q.Threshold), &q.Rounds, &q.TotalRowsRead, &q.StartedAt, &q.CompletedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return out, pgx.ErrNoRows
	}
	if err != nil {
		return out, err
	}
	q.FailureClass = deref(failureClass)
	q.FailureMessage = deref(failureMessage)

	rows, err := s.pool.Query(ctx,
		`SELECT item_id,score FROM topk_query_results WHERE request_id=$1 ORDER BY position`, requestID)
	if err != nil {
		return out, err
	}
	for rows.Next() {
		var k domain.Key
		var id string
		if err := rows.Scan(&id, &k.Score); err != nil {
			rows.Close()
			return out, err
		}
		k.ID = domain.ItemID(id)
		out.Keys = append(out.Keys, k)
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return out, err
	}
	rows.Close()

	statusRows, err := s.pool.Query(ctx, `
SELECT shard_id,exhausted,active,upper_bound_key,cursor_key,pages,rows_read
FROM topk_query_shard_states WHERE request_id=$1 ORDER BY shard_id`, requestID)
	if err != nil {
		return out, err
	}
	for statusRows.Next() {
		var st PersistedStatus
		if err := statusRows.Scan(&st.ID, &st.Exhausted, &st.Active,
			jsonScan(&st.UpperBound), jsonScan(&st.Cursor), &st.Pages, &st.RowsRead); err != nil {
			statusRows.Close()
			return out, err
		}
		out.Statuses = append(out.Statuses, st)
	}
	if err := statusRows.Err(); err != nil {
		statusRows.Close()
		return out, err
	}
	statusRows.Close()

	eventRows, err := s.pool.Query(ctx, `
SELECT kind,shard_id,round,cursor_key,upper_bound_key,threshold_key,item_keys,reason,failure_class,rows_read,occurred_at
FROM topk_query_events WHERE request_id=$1 ORDER BY seq`, requestID)
	if err != nil {
		return out, err
	}
	for eventRows.Next() {
		var ev PersistedEvent
		var shardID *int
		var class *string
		if err := eventRows.Scan(&ev.Kind, &shardID, &ev.Round,
			jsonScan(&ev.Cursor), jsonScan(&ev.UpperBound), jsonScan(&ev.Threshold), jsonScan(&ev.Items),
			&ev.Reason, &class, &ev.RowsRead, &ev.At); err != nil {
			eventRows.Close()
			return out, err
		}
		if shardID != nil {
			ev.ShardID = *shardID
		} else {
			ev.ShardID = -1
		}
		ev.FailureClass = deref(class)
		out.Events = append(out.Events, ev)
	}
	if err := eventRows.Err(); err != nil {
		eventRows.Close()
		return out, err
	}
	eventRows.Close()

	q.Keys = out.Keys
	q.Statuses = out.Statuses
	q.Events = out.Events
	return out, nil
}

func jsonArg(v any) []byte {
	if v == nil {
		return nil
	}
	b, err := json.Marshal(v)
	if err != nil {
		return nil
	}
	return b
}

func jsonScan(v any) scanner {
	return &jsonScanner{target: v}
}

type jsonScanner struct{ target any }

func (s *jsonScanner) Scan(src any) error {
	if src == nil {
		return nil
	}
	b, ok := src.([]byte)
	if !ok {
		t, ok := src.(string)
		if !ok {
			return errors.New("json column is not bytes or text")
		}
		b = []byte(t)
	}
	return json.Unmarshal(b, s.target)
}

func nullNegative(v int) int {
	if v < 0 {
		return -1
	}
	return v
}

func deref(v *string) string {
	if v == nil {
		return ""
	}
	return *v
}

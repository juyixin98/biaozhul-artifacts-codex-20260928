package coordinator

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"time"

	"specbatch/internal/logging"
	"specbatch/internal/model"
	"specbatch/internal/store"
)

var errJobNotComplete = errors.New("job is not complete")

type Runner struct {
	store       *store.Store
	epoch       int64
	tick        time.Duration
	artifactDir string
	log         *logging.Logger
}

func NewRunner(st *store.Store, epoch int64, tick time.Duration, artifactDir string, log *logging.Logger) *Runner {
	return &Runner{store: st, epoch: epoch, tick: tick, artifactDir: artifactDir, log: log}
}

func (r *Runner) Run(ctx context.Context) {
	t := time.NewTicker(r.tick)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case now := <-t.C:
			created, err := r.store.Schedule(ctx, r.epoch, now)
			if err != nil && !errors.Is(err, context.Canceled) {
				r.log.Error("scheduler_tick_failed", map[string]any{"error": err.Error(), "run_epoch": r.epoch})
				continue
			}
			for _, a := range created {
				r.log.Info("scheduler_launch", map[string]any{
					"job_id": a.JobID, "shard_id": a.ShardID, "attempt_no": a.AttemptNo,
					"kind": a.Kind, "run_epoch": r.epoch, "budget_decision": "see attempt_launched event reason",
				})
			}
			r.repairFiles(ctx)
		}
	}
}

func (r *Runner) repairFiles(ctx context.Context) {
	entries, err := os.ReadDir(r.artifactDir)
	if err != nil {
		return
	}
	for _, ent := range entries {
		if !ent.IsDir() {
			name := ent.Name()
			if len(name) > 0 && name[0] == '.' && filepath.Ext(name) == "" {
				_ = os.Remove(filepath.Join(r.artifactDir, name))
			}
		}
	}
	rows, err := r.store.DB().QueryContext(ctx, `SELECT id,status FROM jobs WHERE status IN ('succeeded','failed')`)
	if err != nil {
		return
	}
	type job struct{ id, status string }
	var jobs []job
	for rows.Next() {
		var j job
		if err := rows.Scan(&j.id, &j.status); err == nil {
			jobs = append(jobs, j)
		}
	}
	rows.Close()
	for _, j := range jobs {
		if j.status == model.JobSucceeded {
			if err := srvPublish(r.store, r.artifactDir, j.id); err == nil {
				r.log.Debug("artifacts_repaired", map[string]any{"job_id": j.id, "run_epoch": r.epoch})
			}
		}
	}
}

func (s *Server) PublishArtifacts(ctx context.Context, jobID string) error {
	job, err := s.store.GetJob(ctx, jobID)
	if err != nil {
		return err
	}
	if job.Status != model.JobSucceeded {
		return errJobNotComplete
	}
	return srvPublish(s.store, s.dir, jobID)
}

func (s *Server) PublishOne(ctx context.Context, jobID, shardID string) error {
	content, err := s.store.ArtifactContent(ctx, jobID, shardID)
	if err != nil {
		return err
	}
	return writeFile(filepath.Join(s.dir, jobID, "shards", shardID+".json"), content)
}

func (s *Server) PublishMerge(ctx context.Context, jobID string) error {
	job, err := s.store.GetJob(ctx, jobID)
	if err != nil || job.Status != model.JobSucceeded {
		return err
	}
	merge, err := s.store.Merged(ctx, jobID)
	if err != nil {
		return err
	}
	raw, err := json.MarshalIndent(merge, "", "  ")
	if err != nil {
		return err
	}
	return writeFile(filepath.Join(s.dir, jobID, "merged.json"), raw)
}

func srvPublish(st *store.Store, dir, jobID string) error {
	merge, err := st.Merged(context.Background(), jobID)
	if err != nil {
		return err
	}
	recordsByShard, err := st.ArtifactRows(context.Background(), jobID)
	if err != nil {
		return err
	}
	for shardID, records := range recordsByShard {
		content, err := st.ArtifactContent(context.Background(), jobID, shardID)
		if err != nil {
			return err
		}
		_ = records
		if err := writeFile(filepath.Join(dir, jobID, "shards", shardID+".json"), content); err != nil {
			return err
		}
	}
	raw, err := json.MarshalIndent(merge, "", "  ")
	if err != nil {
		return err
	}
	return writeFile(filepath.Join(dir, jobID, "merged.json"), raw)
}

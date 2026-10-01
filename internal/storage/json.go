package storage

import "opp299/internal/protocol"

func cloneCheckpoint(cp protocol.Checkpoint) protocol.Checkpoint {
	out := cp
	if cp.State != nil {
		out.State = cp.State.Clone()
	}
	out.Positions = make(map[string]protocol.Position, len(cp.Positions))
	for k, v := range cp.Positions {
		out.Positions[k] = v
	}
	out.InFlight = make(map[string][]protocol.InFlight, len(cp.InFlight))
	for stream, items := range cp.InFlight {
		out.InFlight[stream] = append([]protocol.InFlight(nil), items...)
	}
	out.Outputs = append([]protocol.Output(nil), cp.Outputs...)
	return out
}

func checkpointsEqual(a, b protocol.Checkpoint) bool {
	if a.TaskID != b.TaskID || a.ID != b.ID || a.Mode != b.Mode || a.Status != b.Status {
		return false
	}
	if len(a.Positions) != len(b.Positions) || len(a.InFlight) != len(b.InFlight) || len(a.Outputs) != len(b.Outputs) {
		return false
	}
	for stream, pos := range a.Positions {
		if b.Positions[stream] != pos {
			return false
		}
	}
	for stream, items := range a.InFlight {
		if len(items) != len(b.InFlight[stream]) {
			return false
		}
		for i := range items {
			if items[i] != b.InFlight[stream][i] {
				return false
			}
		}
	}
	for key, count := range a.State.Counts {
		if b.State.Counts[key] != count {
			return false
		}
	}
	return true
}

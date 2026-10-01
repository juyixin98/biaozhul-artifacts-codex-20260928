package compute

import "strconv"

type Checkpoint struct {
	ShardID string
	Summary *Summary
}

// MergeTree merges complete shard summaries in the requested association order.
// It is intentionally separate from Merge so tests can compare trees without
// involving storage or HTTP. Empty summaries are legal identity inputs.
func MergeTree(summaries []*Summary, balanced bool) (*Summary, error) {
	out, _, err := MergeTreeWithCheckpoints(summaries, balanced)
	return out, err
}

func MergeTreeWithCheckpoints(summaries []*Summary, balanced bool) (*Summary, []Checkpoint, error) {
	if len(summaries) == 0 {
		return nil, nil, nil
	}
	if !balanced {
		out := Empty(summaries[0].Dim)
		checkpoints := make([]Checkpoint, 0, len(summaries))
		for i, s := range summaries {
			if err := MergeInto(out, s); err != nil {
				return nil, nil, err
			}
			checkpoints = append(checkpoints, Checkpoint{
				ShardID: checkpointLabel(i),
				Summary: Clone(out),
			})
		}
		return out, checkpoints, nil
	}

	type node struct {
		label   string
		summary *Summary
	}
	working := make([]node, len(summaries))
	for i := range summaries {
		working[i] = node{label: checkpointLabel(i), summary: Clone(summaries[i])}
	}

	var checkpoints []Checkpoint
	for len(working) > 1 {
		var next []node
		for i := 0; i < len(working); i += 2 {
			if i+1 == len(working) {
				next = append(next, working[i])
				continue
			}
			merged, err := Merge(working[i].summary, working[i+1].summary)
			if err != nil {
				return nil, nil, err
			}
			label := "(" + working[i].label + "+" + working[i+1].label + ")"
			next = append(next, node{label: label, summary: merged})
			checkpoints = append(checkpoints, Checkpoint{ShardID: label, Summary: Clone(merged)})
		}
		working = next
	}
	if len(checkpoints) == 0 {
		checkpoints = append(checkpoints, Checkpoint{
			ShardID: working[0].label,
			Summary: Clone(working[0].summary),
		})
	}
	checkpoints = append(checkpoints, Checkpoint{
		ShardID: "balanced-tree-root",
		Summary: Clone(working[0].summary),
	})
	return working[0].summary, checkpoints, nil
}

func checkpointLabel(index int) string {
	return "partition-" + strconv.Itoa(index+1)
}

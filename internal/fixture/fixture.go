package fixture

import "topk-demo/internal/protocol"

const Generation = "gen-fixture-001"

type Item struct {
	Generation string
	ShardID    string
	Candidate  protocol.Candidate
	Label      string
}

// Items is an explicit, deterministic local fixture. Tests sort this literal
// slice independently; the top-k kernel never produces the expected answers.
func Items() []Item {
	items := []Item{}
	add := func(shard string, score int64, itemID, label string) {
		items = append(items, Item{
			Generation: Generation,
			ShardID:    shard,
			Candidate:  protocol.Candidate{ItemID: itemID, Score: score},
			Label:      label,
		})
	}

	// Hotspot: shard-a owns a dense prefix and forces same-shard early pruning.
	for _, row := range []struct {
		id    string
		score int64
	}{
		{"a-0001", 1000}, {"a-0002", 995}, {"a-0003", 990}, {"a-0004", 985},
		{"a-0005", 980}, {"a-0006", 900}, {"a-0007", 850}, {"a-0008", 800},
		{"a-0009", 700}, {"a-0010", 600},
	} {
		add("shard-a", row.score, row.id, "synthetic-hotspot")
	}

	// Cross-shard ties: smaller ID wins exactly as the global comparator says.
	add("shard-b", 995, "a-0002-tie", "synthetic-tie-after-a")
	add("shard-b", 990, "0000-b-tie", "synthetic-tie-before-a")
	add("shard-b", 985, "b-0001", "synthetic-upper")
	add("shard-b", 979, "b-0002", "synthetic-just-below")
	add("shard-b", 975, "b-0003", "synthetic-tie-left")
	add("shard-c", 975, "a-0000-tie", "synthetic-tie-before-b")
	add("shard-c", 970, "c-0001", "synthetic-low")
	add("shard-c", 960, "c-0002", "synthetic-low")
	add("shard-c", 950, "c-0003", "synthetic-low")
	add("shard-c", 100, "c-0004", "synthetic-tail")
	add("shard-d", 50, "d-0001", "synthetic-empty-neighbor")

	return items
}

func ShardIDs() []string {
	return []string{"shard-a", "shard-b", "shard-c", "shard-d"}
}

func ExpectedTop5() []protocol.Candidate {
	return []protocol.Candidate{
		{ItemID: "a-0001", Score: 1000},
		{ItemID: "a-0002", Score: 995},
		{ItemID: "a-0002-tie", Score: 995},
		{ItemID: "0000-b-tie", Score: 990},
		{ItemID: "a-0003", Score: 990},
	}
}

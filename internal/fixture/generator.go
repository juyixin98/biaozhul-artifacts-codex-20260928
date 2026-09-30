package fixture

import (
	"fmt"

	"opp297/topk/internal/domain"
	"opp297/topk/internal/storage"
)

// DefaultGeneration is deliberately fixed so local replay can be repeated.
const DefaultGeneration domain.GenerationID = "gen-fixture-v1"

// Default creates four shards with:
// shard 0: a hot concentrated prefix and tied scores,
// shard 1: the same top score and tie boundary,
// shard 2: sparse mid-range rows,
// shard 3: empty.
func Default() []storage.ItemRecord {
	var records []storage.ItemRecord
	add := func(shard int, item string, score int64) {
		records = append(records, storage.ItemRecord{
			Generation: DefaultGeneration,
			ShardID:    shard,
			ItemID:     domain.ItemID(fmt.Sprintf("item-%s", item)),
			Score:      score,
			Label:      fmt.Sprintf("sensitive shard%d/%s", shard, item),
		})
	}

	// Global top six: a00/a01/a02, b00 at 100 ties; a03,b01 at 99.
	add(0, "a00", 100)
	add(0, "a01", 100)
	add(0, "a02", 100)
	add(1, "b00", 100)
	add(0, "a03", 99)
	add(1, "b01", 99)

	add(0, "a04", 80)
	add(0, "a05", 70)
	add(0, "a06", 60)
	add(1, "b02", 85)
	add(1, "b03", 75)
	add(1, "b04", 50)
	add(2, "c00", 90)
	add(2, "c01", 40)
	add(2, "c02", 30)
	return records
}

func ExpectedTopK(k int) []domain.Key {
	all := []domain.Key{
		{Score: 100, ID: "item-a00"},
		{Score: 100, ID: "item-a01"},
		{Score: 100, ID: "item-a02"},
		{Score: 100, ID: "item-b00"},
		{Score: 99, ID: "item-a03"},
		{Score: 99, ID: "item-b01"},
		{Score: 90, ID: "item-c00"},
		{Score: 85, ID: "item-b02"},
		{Score: 80, ID: "item-a04"},
		{Score: 75, ID: "item-b03"},
		{Score: 70, ID: "item-a05"},
		{Score: 60, ID: "item-a06"},
		{Score: 50, ID: "item-b04"},
		{Score: 40, ID: "item-c01"},
		{Score: 30, ID: "item-c02"},
	}
	if k > len(all) {
		k = len(all)
	}
	return append([]domain.Key(nil), all[:k]...)
}

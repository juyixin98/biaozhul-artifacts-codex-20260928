package kernel_test

import (
	"context"
	"errors"
	"sort"
	"testing"
	"time"

	"opp297/topk/internal/domain"
	"opp297/topk/internal/fixture"
	"opp297/topk/internal/kernel"
)

var testSecret = []byte("0123456789abcdef")

type fakeShard struct {
	gen   domain.GenerationID
	items []domain.Candidate
}

type fakeReader struct {
	shards     map[int]fakeShard
	failAfter  map[int]int
	badBoundAt map[int]int
	badOrderAt map[int]int
	wrongGen   map[int]bool
}

func (r *fakeReader) ReadPage(_ context.Context, req domain.PageRequest) (domain.PageResponse, error) {
	shard, ok := r.shards[req.ShardID]
	if !ok {
		return domain.PageResponse{}, domain.ErrUnavailable
	}
	pageNumber := 0
	if req.Cursor != nil {
		pageNumber = int(req.Cursor.Score % 1000) // Unused by the deterministic fixture faults.
	}
	_ = pageNumber

	start := 0
	if req.Cursor != nil {
		start = -1
		for i, item := range shard.items {
			if item.Key().Equal(*req.Cursor) {
				start = i + 1
				break
			}
		}
		if start < 0 {
			return domain.PageResponse{}, errors.New("unknown cursor")
		}
	}

	page := domain.PageResponse{Generation: shard.gen, ShardID: req.ShardID, RowsRead: 0}
	count := 0
	probe := -1
	for i := start; i < len(shard.items) && count < req.Limit; i++ {
		page.Items = append(page.Items, shard.items[i])
		count++
	}
	page.RowsRead = len(page.Items)
	if start+len(page.Items) < len(shard.items) && len(page.Items) == req.Limit {
		probe = start + len(page.Items)
		page.RowsRead++
		page.HasMore = true
		bound := shard.items[probe].Key()
		page.UpperBound = &bound
	}
	if len(page.Items) > 0 {
		cursor := page.Items[len(page.Items)-1].Key()
		page.Cursor = &cursor
	}
	if page.HasMore {
		page.BoundProof = kernel.SignBound(testSecret,
			kernel.BoundPayload(shard.gen, req.ShardID, *page.Cursor, *page.UpperBound))
	}

	pageNo := 0
	if req.Cursor != nil {
		for _, item := range shard.items {
			pageNo++
			if item.Key() == *req.Cursor {
				break
			}
		}
	}
	if want, ok := r.badBoundAt[req.ShardID]; ok && want == pageNo && page.HasMore {
		page.UpperBound.Score--
		page.BoundProof = kernel.SignBound(testSecret,
			kernel.BoundPayload(shard.gen, req.ShardID, *page.Cursor, *page.UpperBound))
	}
	if want, ok := r.badOrderAt[req.ShardID]; ok && want == pageNo && len(page.Items) > 1 {
		page.Items[0], page.Items[1] = page.Items[1], page.Items[0]
	}
	if r.wrongGen[req.ShardID] {
		page.Generation = "other-world"
	}
	if r.failAt(req.ShardID, pageNo) {
		return page, errors.New("simulated participant reset")
	}
	return page, nil
}

func (r *fakeReader) failAt(shard, pageNo int) bool {
	want, ok := r.failAfter[shard]
	return ok && want == pageNo
}

type timeoutReader struct {
	*fakeReader
	timeoutShard int
}

type forgedBoundReader struct {
	*fakeReader
	shard int
}

func (r forgedBoundReader) ReadPage(ctx context.Context, req domain.PageRequest) (domain.PageResponse, error) {
	page, err := r.fakeReader.ReadPage(ctx, req)
	if err == nil && req.ShardID == r.shard && page.HasMore {
		page.UpperBound.Score--
	}
	return page, err
}

func (r timeoutReader) ReadPage(ctx context.Context, req domain.PageRequest) (domain.PageResponse, error) {
	if req.ShardID == r.timeoutShard && req.Cursor == nil {
		return domain.PageResponse{}, context.DeadlineExceeded
	}
	return r.fakeReader.ReadPage(ctx, req)
}

func fixtureReader() *fakeReader {
	byShard := map[int][]domain.Candidate{}
	for _, record := range fixture.Default() {
		byShard[record.ShardID] = append(byShard[record.ShardID],
			domain.Candidate{ID: record.ItemID, Score: record.Score})
	}
	shards := make(map[int]fakeShard, 4)
	for id := 0; id < 4; id++ {
		items := byShard[id]
		shards[id] = fakeShard{gen: fixture.DefaultGeneration, items: items}
	}
	return &fakeReader{
		shards:     shards,
		failAfter:  map[int]int{},
		badBoundAt: map[int]int{},
		badOrderAt: map[int]int{},
		wrongGen:   map[int]bool{},
	}
}

func oracleTopK(k int) []domain.Key {
	var all []domain.Candidate
	for _, record := range fixture.Default() {
		all = append(all, domain.Candidate{ID: record.ItemID, Score: record.Score})
	}
	sort.Slice(all, func(i, j int) bool {
		a, b := all[i].Key(), all[j].Key()
		return a.Better(b)
	})
	if k > len(all) {
		k = len(all)
	}
	out := make([]domain.Key, 0, k)
	for _, item := range all[:k] {
		out = append(out, item.Key())
	}
	return out
}

func TestExactGlobalTopKWithHotspotTiesAndEmptyShard(t *testing.T) {
	cases := []struct {
		k        int
		wantRows int
	}{
		{1, 11}, {5, 15}, {6, 15}, {7, 15}, {8, 15}, {15, 18}, {20, 18},
	}
	for _, tc := range cases {
		k := tc.k
		t.Run("", func(t *testing.T) {
			result, err := kernel.Execute(context.Background(), fixture.DefaultGeneration, kernel.Options{
				ShardIDs:    []int{0, 1, 2, 3},
				K:           k,
				PageSize:    3,
				BoundSecret: testSecret,
			}, fixtureReader())
			if err != nil {
				t.Fatalf("Execute: %v", err)
			}
			if !result.Complete || !result.Exact {
				t.Fatalf("Complete=%v Exact=%v, want both true", result.Complete, result.Exact)
			}
			want := fixture.ExpectedTopK(k)
			if !keysEqual(result.Keys, want) {
				t.Fatalf("keys=%v want=%v", result.Keys, want)
			}
			if result.Keys != nil && len(result.Keys) > 1 {
				for i := 1; i < len(result.Keys); i++ {
					if !result.Keys[i-1].Better(result.Keys[i]) {
						t.Fatalf("result not globally ordered at %d: %v", i, result.Keys)
					}
				}
			}
			if result.TotalRows != tc.wantRows {
				t.Fatalf("rows read=%d want=%d; proof=%+v", result.TotalRows, tc.wantRows, result.Events[len(result.Events)-1])
			}
			if result.Events[len(result.Events)-1].Kind != kernel.EventStop {
				t.Fatalf("last event is %v, want stop proof", result.Events[len(result.Events)-1].Kind)
			}
		})
	}
}

func TestOracleAgreesAcrossAllK(t *testing.T) {
	for k := 1; k <= 20; k++ {
		result, err := kernel.Execute(context.Background(), fixture.DefaultGeneration, kernel.Options{
			ShardIDs: []int{0, 1, 2, 3}, K: k, PageSize: 2, BoundSecret: testSecret,
		}, fixtureReader())
		if err != nil {
			t.Fatalf("k=%d Execute: %v", k, err)
		}
		want := oracleTopK(k)
		if !keysEqual(result.Keys, want) {
			t.Fatalf("k=%d got=%v want=%v", k, result.Keys, want)
		}
	}
}

func TestTimeoutIsIncompleteAndKeepsKnownBounds(t *testing.T) {
	reader := fixtureReader()
	result, err := kernel.Execute(context.Background(), fixture.DefaultGeneration, kernel.Options{
		ShardIDs: []int{0, 1, 2, 3}, K: 5, PageSize: 3,
		Timeout:     time.Millisecond,
		BoundSecret: testSecret,
	}, timeoutReader{fakeReader: reader, timeoutShard: 2})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if result.Complete || result.Exact {
		t.Fatalf("timeout result must be incomplete: %+v", result)
	}
	if result.Failure == nil || result.Failure.Class != domain.FailureTimeout {
		t.Fatalf("failure=%v, want SHARD_TIMEOUT", result.Failure)
	}
	for _, st := range result.Statuses {
		if st.ID == 2 && (st.UpperBound == nil || st.UpperBound.Score != kernel.MaxKeyScore) {
			t.Fatalf("unread timed-out shard must retain the maximum unknown bound, got %+v", st)
		}
	}
}

func TestWrongGenerationIsRejected(t *testing.T) {
	reader := fixtureReader()
	reader.wrongGen[1] = true
	result, err := kernel.Execute(context.Background(), fixture.DefaultGeneration, kernel.Options{
		ShardIDs: []int{0, 1, 2, 3}, K: 3, PageSize: 3, BoundSecret: testSecret,
	}, reader)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if result.Failure == nil || result.Failure.Class != domain.FailureGenerationMismatch {
		t.Fatalf("failure=%v, want generation mismatch", result.Failure)
	}
}

func TestForgedLowerUpperBoundIsRejectedBeforeStop(t *testing.T) {
	reader := fixtureReader()
	result, err := kernel.Execute(context.Background(), fixture.DefaultGeneration, kernel.Options{
		ShardIDs: []int{0, 1, 2, 3}, K: 1, PageSize: 2, BoundSecret: testSecret,
	}, forgedBoundReader{fakeReader: reader, shard: 0})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if result.Failure == nil || result.Failure.Class != domain.FailureBoundViolation {
		t.Fatalf("failure=%v, want bound violation", result.Failure)
	}
	if result.Exact || result.Complete {
		t.Fatalf("bad bound must never be marked exact")
	}
}

func TestBadOrderIsProtocolFailure(t *testing.T) {
	reader := fixtureReader()
	reader.badOrderAt[0] = 0
	result, err := kernel.Execute(context.Background(), fixture.DefaultGeneration, kernel.Options{
		ShardIDs: []int{0, 1, 2, 3}, K: 3, PageSize: 3, BoundSecret: testSecret,
	}, reader)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if result.Failure == nil || result.Failure.Class != domain.FailureProtocol {
		t.Fatalf("failure=%v, want protocol failure", result.Failure)
	}
}

func keysEqual(a, b []domain.Key) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

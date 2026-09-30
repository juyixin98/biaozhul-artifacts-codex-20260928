package topk

import (
	"context"
	"sort"
	"strings"
	"testing"
	"time"

	"topk-demo/internal/protocol"
)

func TestGlobalComparatorAndHeapTieBreaker(t *testing.T) {
	a := protocol.Candidate{ItemID: "a", Score: 10}
	b := protocol.Candidate{ItemID: "b", Score: 10}
	c := protocol.Candidate{ItemID: "z", Score: 11}
	if !protocol.Better(a, b) || !protocol.Better(c, a) {
		t.Fatalf("global comparator has wrong score/tie ordering")
	}
	h := NewTopHeap(2)
	h.Add(b)
	h.Add(a)
	if added := h.Add(c); !added {
		t.Fatalf("better candidate must displace kth item")
	}
	if added := h.Add(protocol.Candidate{ItemID: "c", Score: 10}); added {
		t.Fatalf("equal-or-worse key must not displace kth item")
	}
	got := h.Sorted()
	want := []protocol.Candidate{c, a}
	assertCandidates(t, "top2", got, want)
}

func TestExactGlobalTopKHotspotTiesEmptyAndEarlyPrune(t *testing.T) {
	data := map[string][]protocol.Candidate{
		"a": {
			{ItemID: "a-0001", Score: 1000},
			{ItemID: "a-0002", Score: 995},
			{ItemID: "a-0003", Score: 990},
			{ItemID: "a-0004", Score: 985},
			{ItemID: "a-0005", Score: 980},
			{ItemID: "a-0006", Score: 900},
		},
		"b": {
			{ItemID: "a-0002-tie", Score: 995},
			{ItemID: "b-0001", Score: 985},
			{ItemID: "b-0002", Score: 979},
		},
		"c": {
			{ItemID: "0000-b-tie", Score: 990},
			{ItemID: "a-0000-tie", Score: 975},
			{ItemID: "c-0001", Score: 970},
		},
		"d": {},
	}
	resp := executeTestQuery(t, data, []string{"a", "b", "c", "d"}, 5, 2, time.Second)
	want := oracle(data, 5)
	assertCandidates(t, "top5", resp.Results, want)
	if !resp.Exact || !resp.Complete || resp.StopReason != protocol.ReasonExactThreshold {
		t.Fatalf("query should be exact: exact=%v complete=%v reason=%s", resp.Exact, resp.Complete, resp.StopReason)
	}
	if resp.Threshold == nil || resp.Threshold.Score != 990 || resp.Threshold.ItemID != "a-0003" {
		t.Fatalf("unexpected threshold: %+v", resp.Threshold)
	}
	proofs := mapProofs(resp.Proofs)
	if proofs["a"].State != StatePruned || proofs["a"].StopUpper == nil {
		t.Fatalf("hotspot shard should be pruned with a concrete upper: %+v", proofs["a"])
	}
	if proofs["d"].State != StateExhausted || !proofs["d"].CursorExhausted {
		t.Fatalf("empty shard must be marked exhausted: %+v", proofs["d"])
	}
	allItems := countAll(data)
	if resp.Metrics.StorageRowsRead >= int64(allItems) {
		t.Fatalf("early termination should read fewer than all rows: got %d all %d", resp.Metrics.StorageRowsRead, allItems)
	}
	if resp.Metrics.CandidatesReturned >= int64(allItems) {
		t.Fatalf("candidate return count indicates no pruning: got %d all %d", resp.Metrics.CandidatesReturned, allItems)
	}
}

func TestTimeoutIsIncompleteAndPreservesKnownUpper(t *testing.T) {
	data := map[string][]protocol.Candidate{
		"good": {{ItemID: "g1", Score: 100}},
		"slow": {{ItemID: "s1", Score: 200}, {ItemID: "s2", Score: 199}, {ItemID: "s3", Score: 198}},
	}
	client := newFakeClient(data)
	client["slow"].scanDelay = 50 * time.Millisecond
	engine := NewEngine(client)
	resp, err := engine.Execute(context.Background(), []string{"good", "slow"}, Options{
		RunID: "run-timeout", Generation: "g1", K: 1, PageSize: 2, ShardTimeout: 5 * time.Millisecond, Sink: NopDiagnostics{},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Exact || resp.Complete || resp.StopReason != protocol.ReasonIncomplete {
		t.Fatalf("timeout query must be incomplete: %+v", resp)
	}
	if len(resp.Errors) != 1 || resp.Errors[0].ShardID != "slow" || resp.Errors[0].Category != protocol.FailureTimeout {
		t.Fatalf("expected timeout category, got %+v", resp.Errors)
	}
	if proofs := mapProofs(resp.Proofs); proofs["slow"].FailureCategory != protocol.FailureTimeout {
		t.Fatalf("proof must preserve timeout category: %+v", proofs["slow"])
	}
}

func TestWrongUpperIsRejectedAsInvalidShard(t *testing.T) {
	data := map[string][]protocol.Candidate{
		"bad": {{ItemID: "real-1", Score: 100}, {ItemID: "real-2", Score: 90}, {ItemID: "real-3", Score: 80}},
	}
	client := newFakeClient(data)
	client["bad"].scanUpperBogus = &protocol.Candidate{ItemID: "fake-upper", Score: 95}
	engine := NewEngine(client)
	resp, err := engine.Execute(context.Background(), []string{"bad"}, Options{
		RunID: "run-bad-upper", Generation: "g1", K: 1, PageSize: 2, ShardTimeout: time.Second, Sink: NopDiagnostics{},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Exact || resp.Complete {
		t.Fatalf("invalid upper response cannot yield exact results")
	}
	if len(resp.Errors) != 1 || resp.Errors[0].Category != protocol.FailureInvalidShard {
		t.Fatalf("expected invalid shard category, got %+v", resp.Errors)
	}
	if !strings.Contains(resp.Errors[0].Message, "ordered after page") {
		t.Fatalf("error should explain next-bound rejection, got %q", resp.Errors[0].Message)
	}
	proofs := mapProofs(resp.Proofs)
	if proofs["bad"].StopUpper == nil || proofs["bad"].StopUpper.ItemID != "fake-upper" {
		t.Fatalf("rejected known upper must be retained in proof: %+v", proofs["bad"])
	}
}

func TestBoundGenerationMismatchIsRejected(t *testing.T) {
	data := map[string][]protocol.Candidate{"old": {{ItemID: "old-1", Score: 100}}}
	client := newFakeClient(data)
	client["old"].scanGeneration = "g-stale"
	engine := NewEngine(client)
	resp, err := engine.Execute(context.Background(), []string{"old"}, Options{
		RunID: "run-generation", Generation: "g-current", K: 1, PageSize: 2, ShardTimeout: time.Second, Sink: NopDiagnostics{},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Exact || len(resp.Errors) != 1 || resp.Errors[0].Category != protocol.FailureInvalidShard {
		t.Fatalf("generation mismatch must reject shard and result: %+v", resp)
	}
}

func TestKExceedsTotalIsExactByExhaustion(t *testing.T) {
	data := map[string][]protocol.Candidate{
		"a": {{ItemID: "a1", Score: 1}},
		"b": {},
	}
	resp := executeTestQuery(t, data, []string{"a", "b"}, 5, 1, time.Second)
	if !resp.Exact || resp.StopReason != protocol.ReasonExactExhausted || len(resp.Results) != 1 {
		t.Fatalf("all-shard exhaustion should give exact partial result: %+v", resp)
	}
}

func executeTestQuery(t *testing.T, data map[string][]protocol.Candidate, ids []string, k, page int, timeout time.Duration) *protocol.QueryResponse {
	t.Helper()
	resp, err := NewEngine(newFakeClient(data)).Execute(context.Background(), ids, Options{
		RunID: "run-test", Generation: "g1", K: k, PageSize: page, ShardTimeout: timeout, Sink: NopDiagnostics{},
	})
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func oracle(data map[string][]protocol.Candidate, k int) []protocol.Candidate {
	var all []protocol.Candidate
	for _, items := range data {
		all = append(all, items...)
	}
	sort.Slice(all, func(i, j int) bool { return protocol.Better(all[i], all[j]) })
	if len(all) > k {
		all = all[:k]
	}
	return all
}

func countAll(data map[string][]protocol.Candidate) int {
	n := 0
	for _, items := range data {
		n += len(items)
	}
	return n
}

func mapProofs(proofs []protocol.ShardProof) map[string]protocol.ShardProof {
	out := map[string]protocol.ShardProof{}
	for _, proof := range proofs {
		out[proof.ShardID] = proof
	}
	return out
}

func assertCandidates(t *testing.T, name string, got, want []protocol.Candidate) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d (got %#v want %#v)", name, len(got), len(want), got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s[%d] = %+v, want %+v; full got %#v", name, i, got[i], want[i], got)
		}
	}
}

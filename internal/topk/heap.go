package topk

import (
	"container/heap"
	"sort"

	"topk-demo/internal/protocol"
)

type TopHeap struct {
	items []protocol.Candidate
	k     int
}

func NewTopHeap(k int) *TopHeap {
	if k <= 0 {
		return &TopHeap{k: k}
	}
	return &TopHeap{items: make([]protocol.Candidate, 0, k), k: k}
}

func (h *TopHeap) Len() int { return len(h.items) }

func (h *TopHeap) K() int { return h.k }

// The current kth-best item sits at the root. A candidate must be strictly
// Better than that root to displace it; an equal global key never displaces it.
func (h *TopHeap) Less(i, j int) bool {
	return protocol.Better(h.items[j], h.items[i])
}

func (h *TopHeap) Swap(i, j int) { h.items[i], h.items[j] = h.items[j], h.items[i] }

func (h *TopHeap) Push(x any) { h.items = append(h.items, x.(protocol.Candidate)) }

func (h *TopHeap) Pop() any {
	old := h.items
	n := len(old)
	item := old[n-1]
	h.items = old[:n-1]
	return item
}

func (h *TopHeap) Add(c protocol.Candidate) bool {
	if h.k <= 0 {
		return false
	}
	if h.items == nil {
		h.items = make([]protocol.Candidate, 0, h.k)
	}
	if len(h.items) < h.k {
		heap.Push(h, c)
		return true
	}
	worst := h.items[0]
	if !protocol.Better(c, worst) {
		return false
	}
	h.items[0] = c
	heap.Fix(h, 0)
	return true
}

func (h *TopHeap) Threshold() *protocol.Candidate {
	if len(h.items) == 0 {
		return nil
	}
	c := h.items[0]
	return &c
}

func (h *TopHeap) Sorted() []protocol.Candidate {
	out := append([]protocol.Candidate(nil), h.items...)
	sort.Slice(out, func(i, j int) bool { return protocol.Better(out[i], out[j]) })
	return out
}

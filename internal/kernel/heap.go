package kernel

import (
	"sort"

	"opp297/topk/internal/domain"
)

// minHeap keeps the worst currently selected global candidate at the top.
type minHeap struct {
	keys []domain.Key
}

func newMinHeap(capacity int) *minHeap {
	return &minHeap{keys: make([]domain.Key, 0, capacity)}
}

func (h *minHeap) Len() int { return len(h.keys) }

func (h *minHeap) Push(k domain.Key) {
	h.keys = append(h.keys, k)
	h.up(len(h.keys) - 1)
}

func (h *minHeap) Pop() domain.Key {
	n := len(h.keys)
	root := h.keys[0]
	last := n - 1
	h.keys[0] = h.keys[last]
	h.keys = h.keys[:last]
	if n > 1 {
		h.down(0)
	}
	return root
}

func (h *minHeap) Peek() domain.Key { return h.keys[0] }

func (h *minHeap) Sorted() []domain.Key {
	out := append([]domain.Key(nil), h.keys...)
	sort.Slice(out, func(i, j int) bool { return out[i].Better(out[j]) })
	return out
}

func (h *minHeap) up(i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if !h.keys[parent].Better(h.keys[i]) {
			return
		}
		h.keys[i], h.keys[parent] = h.keys[parent], h.keys[i]
		i = parent
	}
}

func (h *minHeap) down(i int) { siftDown(h.keys, i, len(h.keys)) }

func siftDown(keys []domain.Key, i, n int) {
	for {
		left := 2*i + 1
		if left >= n {
			return
		}
		child := left
		right := left + 1
		if right < n && keys[left].Better(keys[right]) {
			child = right
		}
		if !keys[i].Better(keys[child]) {
			return
		}
		keys[i], keys[child] = keys[child], keys[i]
		i = child
	}
}

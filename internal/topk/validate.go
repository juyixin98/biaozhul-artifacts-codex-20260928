package topk

import (
	"fmt"

	"topk-demo/internal/protocol"
)

func validateBoundResponse(generation, shardID string, resp *protocol.BoundResponse) error {
	if resp == nil {
		return fmt.Errorf("empty bound response")
	}
	if resp.Generation != generation {
		return fmt.Errorf("generation mismatch: got %q want %q", resp.Generation, generation)
	}
	if resp.ShardID != shardID {
		return fmt.Errorf("shard mismatch: got %q want %q", resp.ShardID, shardID)
	}
	if resp.Exhausted {
		if resp.Upper != nil {
			return fmt.Errorf("exhausted bound cannot carry an upper candidate")
		}
		if resp.ReadRows != 0 || resp.BoundProbes != 1 {
			return fmt.Errorf("exhausted bound metrics are invalid")
		}
		return nil
	}
	if resp.Upper == nil {
		return fmt.Errorf("non-exhausted bound must carry upper candidate")
	}
	if err := protocol.ValidateCandidate(*resp.Upper); err != nil {
		return err
	}
	if resp.ReadRows < 1 || resp.BoundProbes != 1 {
		return fmt.Errorf("successful bound must read one row and record one probe")
	}
	return nil
}

func validateScanResponse(generation, shardID, cursor string, pageSize int, knownUpper *protocol.Candidate, resp *protocol.ScanResponse, initial bool) error {
	if resp == nil {
		return fmt.Errorf("empty scan response")
	}
	if resp.Generation != generation {
		return fmt.Errorf("generation mismatch: got %q want %q", resp.Generation, generation)
	}
	if resp.ShardID != shardID {
		return fmt.Errorf("shard mismatch: got %q want %q", resp.ShardID, shardID)
	}
	if resp.Cursor != cursor {
		return fmt.Errorf("cursor echo mismatch")
	}
	if pageSize <= 0 || len(resp.Items) > pageSize {
		return fmt.Errorf("page size violated")
	}
	if resp.Items == nil {
		return fmt.Errorf("items must be a JSON array")
	}
	if len(resp.Items) == 0 {
		if !resp.Exhausted || resp.NextUpper != nil || resp.NextCursor != "" {
			return fmt.Errorf("empty page must be terminal and cannot carry a next upper")
		}
		if resp.ReadRows != 0 || resp.ReturnedRows != 0 || resp.BoundProbes != 0 {
			return fmt.Errorf("empty page metrics are invalid")
		}
		return nil
	}

	if initial {
		if cursor != "" || knownUpper != nil || resp.ItemsIncludedUpper {
			return fmt.Errorf("initial scan must not resume a prior upper")
		}
	} else {
		if knownUpper == nil || resp.RequestedUpper == nil || *knownUpper != *resp.RequestedUpper {
			return fmt.Errorf("scan must echo the requested exact upper bound")
		}
		if !resp.ItemsIncludedUpper {
			return fmt.Errorf("scan must report that returned items include the requested upper")
		}
		if *knownUpper != resp.Items[0] {
			return fmt.Errorf("first scanned item does not equal the exact upper bound")
		}
	}
	if resp.ReturnedRows != int64(len(resp.Items)) {
		return fmt.Errorf("returned_rows does not match item count")
	}

	seen := make(map[string]struct{}, len(resp.Items))
	for i := range resp.Items {
		if err := protocol.ValidateCandidate(resp.Items[i]); err != nil {
			return err
		}
		key := protocol.CandidateKey(&resp.Items[i])
		if _, ok := seen[key]; ok {
			return fmt.Errorf("duplicate candidate in scan page: %s", key)
		}
		seen[key] = struct{}{}
		if i > 0 && !protocol.Better(resp.Items[i-1], resp.Items[i]) {
			return fmt.Errorf("scan items are not strictly globally ordered")
		}
	}

	last := resp.Items[len(resp.Items)-1]
	if resp.Exhausted {
		if resp.NextUpper != nil || resp.NextCursor != "" || resp.NextCursorAt != nil {
			return fmt.Errorf("exhausted page cannot carry next upper or next cursor")
		}
		if resp.BoundProbes != 0 || resp.ReadRows != int64(len(resp.Items)) {
			return fmt.Errorf("exhausted page metrics are invalid")
		}
		return nil
	}

	if len(resp.Items) != pageSize {
		return fmt.Errorf("non-terminal page must contain exactly page_size items")
	}
	if resp.NextUpper == nil || resp.NextCursor == "" || resp.NextCursorAt == nil {
		return fmt.Errorf("non-terminal page must carry next upper and next cursor")
	}
	if *resp.NextCursorAt != *resp.NextUpper {
		return fmt.Errorf("next cursor must point exactly at the proven next upper")
	}
	if err := protocol.ValidateCandidate(*resp.NextUpper); err != nil {
		return err
	}
	if _, ok := seen[protocol.CandidateKey(resp.NextUpper)]; ok {
		return fmt.Errorf("next upper must be beyond the returned page")
	}
	if !protocol.Better(last, *resp.NextUpper) {
		return fmt.Errorf("next upper is not ordered after page")
	}
	expectedReadRows := int64(len(resp.Items))
	if !initial {
		expectedReadRows++
	}
	if resp.ReadRows != expectedReadRows || resp.BoundProbes != 1 {
		return fmt.Errorf("non-terminal page must report exact storage reads and one probe")
	}
	return nil
}

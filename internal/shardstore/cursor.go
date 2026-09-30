package shardstore

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
)

type Cursor struct {
	ResumeScore  int64  `json:"resume_score"`
	ResumeItemID string `json:"resume_item_id"`
	Include      bool   `json:"include"`
	Version      int    `json:"version"`
}

func EncodeCursor(c Cursor) string {
	raw, _ := json.Marshal(c)
	return base64.RawURLEncoding.EncodeToString(raw)
}

func DecodeCursor(s string) (Cursor, error) {
	var c Cursor
	if s == "" {
		return Cursor{Version: 1}, nil
	}
	raw, err := base64.RawURLEncoding.DecodeString(s)
	if err != nil {
		return Cursor{}, fmt.Errorf("decode cursor: %w", err)
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return Cursor{}, fmt.Errorf("unmarshal cursor: %w", err)
	}
	if c.Version != 1 {
		return Cursor{}, fmt.Errorf("unsupported cursor version %d", c.Version)
	}
	if c.ResumeItemID == "" && (c.ResumeScore != 0 || s != "") {
		return Cursor{}, fmt.Errorf("non-initial cursor must carry after_item_id")
	}
	return c, nil
}

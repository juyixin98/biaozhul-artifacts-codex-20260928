package kernel

import (
	"crypto/hmac"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"fmt"

	"opp297/topk/internal/domain"
)

func BoundPayload(gen domain.GenerationID, shardID int, cursor, bound domain.Key) string {
	return fmt.Sprintf("%s|%d|%d|%s|%d|%s",
		gen, shardID, cursor.Score, cursor.ID, bound.Score, bound.ID)
}

func SignBound(secret []byte, payload string) string {
	mac := hmac.New(sha256.New, secret)
	_, _ = mac.Write([]byte(payload))
	return base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
}

func VerifyBound(secret []byte, gen domain.GenerationID, shardID int, cursor, bound domain.Key, signature string) bool {
	if len(secret) == 0 || signature == "" {
		return false
	}
	want := SignBound(secret, BoundPayload(gen, shardID, cursor, bound))
	return subtle.ConstantTimeCompare([]byte(want), []byte(signature)) == 1
}

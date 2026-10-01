package id

import (
	"crypto/rand"
	"encoding/hex"

	"diststats/internal/apperror"
)

func New() (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", apperror.Unavailable("RANDOM_ID_FAILED", "cannot generate identifier", err)
	}
	// RFC 4122 variant and version for UUIDv4, kept easy for PostgreSQL/JSON.
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	dst := make([]byte, 36)
	hex.Encode(dst[0:8], b[0:4])
	dst[8] = '-'
	hex.Encode(dst[9:13], b[4:6])
	dst[13] = '-'
	hex.Encode(dst[14:18], b[6:8])
	dst[18] = '-'
	hex.Encode(dst[19:23], b[8:10])
	dst[23] = '-'
	hex.Encode(dst[24:36], b[10:16])
	return string(dst), nil
}

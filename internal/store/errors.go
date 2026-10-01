package store

import "errors"

var (
	ErrNotFound     = errors.New("not found")
	ErrInvalidState = errors.New("invalid state")
	ErrFenced       = errors.New("coordinator epoch fenced")
	ErrChecksum     = errors.New("invalid artifact checksum")
	ErrNoRowsLocked = errors.New("no rows locked")
)

package service

import "errors"

var (
	ErrUnknownGeneration = errors.New("unknown generation")
	ErrUnknownRequest    = errors.New("unknown request")
)

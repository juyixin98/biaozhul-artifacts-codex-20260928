package coordinator

import "time"

func durationMS(ms int) time.Duration {
	return time.Duration(ms) * time.Millisecond
}

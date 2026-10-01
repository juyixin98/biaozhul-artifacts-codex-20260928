package main

import (
	"context"
	"log"
	"net/http"
	"time"

	"checkpointlab/internal/api"
	"checkpointlab/internal/config"
	"checkpointlab/internal/storage"
)

func main() {
	cfg := config.Load()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	var store storage.Store
	if cfg.StoreKind == "postgres" {
		pg, err := storage.OpenPostgres(ctx, cfg.DatabaseURL)
		if err != nil {
			log.Fatalf("connect postgres: %v", err)
		}
		store = pg
	} else {
		store = storage.NewMemoryStore()
	}

	app := api.New(store)
	server := &http.Server{
		Addr:              cfg.Addr,
		Handler:           app.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
	}
	log.Printf("checkpointlab listening on %s store=%s", cfg.Addr, cfg.StoreKind)
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
}

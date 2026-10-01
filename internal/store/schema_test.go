package store

import (
	"os"
	"testing"
)

func TestEmbeddedSchemaMatchesMigrationFile(t *testing.T) {
	fileSchema, err := os.ReadFile("../../migrations/0001_init.sql")
	if err != nil {
		t.Fatal(err)
	}
	if string(fileSchema) != migration {
		t.Fatal("embedded schema.sql and migrations/0001_init.sql differ")
	}
}

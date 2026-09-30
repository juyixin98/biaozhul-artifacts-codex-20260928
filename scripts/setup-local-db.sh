#!/usr/bin/env bash
set -euo pipefail

DB_NAME="${DB_NAME:-opp297_topk}"
PSQL_USER="${PSQL_USER:-$USER}"

if ! psql -h /var/run/postgresql -U "$PSQL_USER" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'" | grep -q 1; then
  if createdb -h /var/run/postgresql -U "$PSQL_USER" "$DB_NAME" 2>/dev/null; then
    :
  else
    sudo -u postgres createdb "$DB_NAME"
    sudo -u postgres psql -d "$DB_NAME" -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $PSQL_USER" >/dev/null
    sudo -u postgres psql -d "$DB_NAME" -c "GRANT ALL ON SCHEMA public TO $PSQL_USER" >/dev/null
  fi
fi
echo "PostgreSQL database $DB_NAME is ready."

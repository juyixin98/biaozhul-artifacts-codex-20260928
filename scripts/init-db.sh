#!/usr/bin/env bash
set -euo pipefail

# Local development/test database. Uses the local PostgreSQL superuser and
# does not require any cloud account or production credentials.
DB_NAME="${DB_NAME:-opp298}"
DB_USER="${DB_USER:-$USER}"

psql_superuser() {
  if sudo -n true 2>/dev/null; then
    sudo -n -u postgres psql -v ON_ERROR_STOP=1 "$@"
  else
    psql -d postgres -v ON_ERROR_STOP=1 "$@"
  fi
}

psql_superuser <<SQL
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${DB_USER}') THEN
    CREATE ROLE ${DB_USER} LOGIN;
  END IF;
END
\$\$;

SELECT 'CREATE DATABASE ${DB_NAME} OWNER ${DB_USER}'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${DB_NAME}')\gexec
SQL

echo "PostgreSQL database ${DB_NAME} is ready for user ${DB_USER}"

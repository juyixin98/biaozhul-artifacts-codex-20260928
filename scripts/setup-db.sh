#!/usr/bin/env bash
set -euo pipefail

ADMIN_PSQL="${ADMIN_PSQL:-sudo -n -u postgres psql}"
APP_USER="${APP_USER:-admin}"

$ADMIN_PSQL -v ON_ERROR_STOP=1 <<SQL
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${APP_USER}') THEN
    CREATE ROLE ${APP_USER} LOGIN;
  END IF;
END
\$\$;
SELECT 'created or verified app role: ${APP_USER}';
SQL

if [[ "$#" -eq 0 ]]; then
  set -- opp300_dev opp300_test
fi

for name in "$@"; do
  $ADMIN_PSQL -v ON_ERROR_STOP=1 -c "DROP DATABASE IF EXISTS ${name};"
  $ADMIN_PSQL -v ON_ERROR_STOP=1 -c "CREATE DATABASE ${name} OWNER ${APP_USER};"
done

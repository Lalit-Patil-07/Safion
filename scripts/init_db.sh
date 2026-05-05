#!/usr/bin/env bash
# scripts/init_db.sh
#
# Runs inside the PostgreSQL container via /docker-entrypoint-initdb.d/
# at first-start only (when the data directory is empty).
#
# The postgres image already creates POSTGRES_USER / POSTGRES_DB from
# environment variables — this script only needs to enable pgvector.
set -euo pipefail

psql -v ON_ERROR_STOP=1 \
     --username "$POSTGRES_USER" \
     --dbname   "$POSTGRES_DB" \
     <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS vector;
EOSQL

echo "pgvector extension enabled in '$POSTGRES_DB'."
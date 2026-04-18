#!/usr/bin/env bash
# scripts/setup_db.sh
#
# Creates the PostgreSQL user, database, grants privileges, and enables pgvector.
# .env is the ONLY source of configuration — no defaults, no fallbacks.
# Fully idempotent — safe to run multiple times.
#
# Usage:
#   bash scripts/setup_db.sh
#
# Superuser authentication:
#   Default : sudo -u postgres  (peer auth, Unix socket — no password required)
#   Override: PGPASSWORD=<pw> bash scripts/setup_db.sh  (TCP password auth)
#
# Requirements:
#   - .env exists at project root with all DB_* variables set
#   - PostgreSQL 18 running
#   - pgvector installed at system level — see scripts/install_postgres.sh
#   - psql in PATH

set -euo pipefail

# ── Locate .env ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: .env not found at $ENV_FILE"
    echo "       Copy .env.example to .env and fill in all required values."
    exit 1
fi

# ── Read a single variable from .env ─────────────────────────────────────────
# Handles: optional quotes, spaces around '=', inline # comments.
read_env_var() {
    local key="$1"
    grep -E "^[[:space:]]*${key}[[:space:]]*=" "$ENV_FILE" \
    | head -1 \
    | sed -E "
        s/^[[:space:]]*${key}[[:space:]]*=[[:space:]]*//
        s/[[:space:]]*#.*\$//" \
    | sed -E "s/^['\"]//; s/['\"]$//; s/[[:space:]]*\$//"
}

# ── Read explicit DB variables ────────────────────────────────────────────────
DB_USER=$(read_env_var DB_USER)
DB_PASSWORD=$(read_env_var DB_PASSWORD)
DB_HOST=$(read_env_var DB_HOST)
DB_PORT=$(read_env_var DB_PORT)
DB_NAME=$(read_env_var DB_NAME)

# ── Validate all variables are present ───────────────────────────────────────
MISSING=""
[[ -z "$DB_USER"     ]] && MISSING="$MISSING DB_USER"
[[ -z "$DB_PASSWORD" ]] && MISSING="$MISSING DB_PASSWORD"
[[ -z "$DB_HOST"     ]] && MISSING="$MISSING DB_HOST"
[[ -z "$DB_PORT"     ]] && MISSING="$MISSING DB_PORT"
[[ -z "$DB_NAME"     ]] && MISSING="$MISSING DB_NAME"

if [[ -n "$MISSING" ]]; then
    echo "ERROR: Missing required database configuration in .env"
    echo "       Missing:$MISSING"
    exit 1
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo "==> Database setup"
echo "    Host:     $DB_HOST:$DB_PORT"
echo "    Database: $DB_NAME"
echo "    User:     $DB_USER"

# ── Superuser psql helper ─────────────────────────────────────────────────────
if [[ -n "${PGPASSWORD:-}" ]]; then
    pg() {
        PGPASSWORD="$PGPASSWORD" psql -v ON_ERROR_STOP=1 \
            -h "$DB_HOST" -p "$DB_PORT" -U postgres "$@"
    }
    echo "    Auth:     PGPASSWORD (TCP)"
else
    pg() {
        sudo -u postgres psql -v ON_ERROR_STOP=1 "$@"
    }
    echo "    Auth:     sudo peer (Unix socket)"
fi
echo ""

# ── Escape single quotes in password for SQL literal ─────────────────────────
ESCAPED_PASSWORD="${DB_PASSWORD//\'/\'\'}"

# ── Create user ───────────────────────────────────────────────────────────────
echo "==> Creating user '$DB_USER'..."
pg -tc "SELECT 1 FROM pg_roles WHERE rolname = '$DB_USER'" \
    | grep -q 1 \
    && echo "    already exists — skipping" \
    || pg -c "CREATE USER $DB_USER WITH PASSWORD '$ESCAPED_PASSWORD';"

# ── Create database ───────────────────────────────────────────────────────────
echo "==> Creating database '$DB_NAME'..."
pg -tc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME'" \
    | grep -q 1 \
    && echo "    already exists — skipping" \
    || pg -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;"

# ── Grant privileges ──────────────────────────────────────────────────────────
echo "==> Granting privileges..."
pg -d "$DB_NAME" -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"
pg -d "$DB_NAME" -c "GRANT ALL ON SCHEMA public TO $DB_USER;"

# ── Enable pgvector ───────────────────────────────────────────────────────────
echo "==> Enabling pgvector extension..."
pg -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS vector;"

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "==> Installed extensions in '$DB_NAME':"
pg -d "$DB_NAME" -c "\dx"

echo ""
echo "==> Setup complete."
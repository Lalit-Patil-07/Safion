#!/usr/bin/env bash
# scripts/setup_db.sh
#
# Creates the PostgreSQL user, database, grants privileges, and enables pgvector.
# Reads DATABASE_URL from .env at the project root.
# Fully idempotent — safe to run multiple times.
#
# Usage:
#   bash scripts/setup_db.sh
#
# Authentication for the postgres superuser:
#   Default : sudo -u postgres (peer auth via Unix socket — works on most Linux installs)
#   Override: set PGPASSWORD env var to use TCP password auth instead
#     PGPASSWORD=secret bash scripts/setup_db.sh
#
# Requirements:
#   - PostgreSQL 18 running locally
#   - pgvector installed at system level (not pip) — see README
#   - psql available in PATH

set -euo pipefail

# ── Locate .env ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: .env not found at $ENV_FILE"
    echo "       Copy .env.example to .env and set your values."
    exit 1
fi

# ── Parse DATABASE_URL from .env ──────────────────────────────────────────────
# Handles: quoted values, spaces around =, inline comments.
# Example formats accepted:
#   DATABASE_URL=postgresql://user:pass@localhost:5432/db
#   DATABASE_URL="postgresql://user:pass@localhost:5432/db"
#   DATABASE_URL = 'postgresql://user:pass@localhost:5432/db'  # comment
DATABASE_URL=$(
    grep -E '^[[:space:]]*DATABASE_URL[[:space:]]*=' "$ENV_FILE" \
    | head -1 \
    | sed -E "
        s/^[[:space:]]*DATABASE_URL[[:space:]]*=[[:space:]]*//
        s/[[:space:]]*#.*\$//
        s/^['\"]//" \
    | sed -E "s/['\"]\$//" \
    | sed -E "s/[[:space:]]*\$//"
)

if [[ -z "$DATABASE_URL" ]]; then
    echo "ERROR: DATABASE_URL not found or empty in .env"
    exit 1
fi

# ── Parse URL components ──────────────────────────────────────────────────────
# Strip scheme (postgresql:// or postgres://)
without_scheme="${DATABASE_URL#*://}"

# Split userinfo from hostinfo at the LAST '@'
# so passwords containing '@' are handled correctly.
userinfo="${without_scheme%@*}"
hostinfo="${without_scheme##*@}"

# User is everything before the first ':' in userinfo
DB_USER="${userinfo%%:*}"
# Password is everything after the first ':' in userinfo
DB_PASSWORD="${userinfo#*:}"

# Host is everything before ':' in hostinfo
DB_HOST="${hostinfo%%:*}"
port_and_db="${hostinfo#*:}"
DB_PORT="${port_and_db%%/*}"
DB_NAME="${port_and_db#*/}"
# Strip any query string from DB_NAME
DB_NAME="${DB_NAME%%\?*}"

echo "==> Database setup"
echo "    Host:     $DB_HOST:$DB_PORT"
echo "    Database: $DB_NAME"
echo "    User:     $DB_USER"
echo ""

# ── Helper: run psql as the postgres superuser ────────────────────────────────
# Default: peer auth via Unix socket using sudo (no postgres password needed).
# Override: set PGPASSWORD in environment to use TCP password auth instead.
if [[ -n "${PGPASSWORD:-}" ]]; then
    pg() {
        PGPASSWORD="$PGPASSWORD" psql \
            -v ON_ERROR_STOP=1 \
            -h "$DB_HOST" -p "$DB_PORT" \
            -U postgres \
            "$@"
    }
    echo "    Auth:     PGPASSWORD (TCP)"
else
    pg() {
        sudo -u postgres psql \
            -v ON_ERROR_STOP=1 \
            "$@"
    }
    echo "    Auth:     sudo peer (Unix socket)"
fi
echo ""

# ── Escape single quotes in password for SQL literal ─────────────────────────
# SQL standard: '' is an escaped single quote inside a single-quoted string.
ESCAPED_PASSWORD="${DB_PASSWORD//\'/\'\'}"

# ── Create user (idempotent) ──────────────────────────────────────────────────
echo "==> Creating user '$DB_USER'..."
pg -tc "SELECT 1 FROM pg_roles WHERE rolname = '$DB_USER'" \
    | grep -q 1 \
    && echo "    already exists — skipping" \
    || pg -c "CREATE USER $DB_USER WITH PASSWORD '$ESCAPED_PASSWORD';"

# ── Create database (idempotent) ──────────────────────────────────────────────
echo "==> Creating database '$DB_NAME'..."
pg -tc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME'" \
    | grep -q 1 \
    && echo "    already exists — skipping" \
    || pg -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;"

# ── Grant privileges ──────────────────────────────────────────────────────────
echo "==> Granting privileges..."
pg -d "$DB_NAME" -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"
pg -d "$DB_NAME" -c "GRANT ALL ON SCHEMA public TO $DB_USER;"

# ── Enable pgvector extension ─────────────────────────────────────────────────
echo "==> Enabling pgvector extension..."
pg -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS vector;"

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "==> Installed extensions in '$DB_NAME':"
pg -d "$DB_NAME" -c "\dx"

echo ""
echo "==> Setup complete."

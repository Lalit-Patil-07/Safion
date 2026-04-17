#!/usr/bin/env bash
# scripts/install_postgres.sh
#
# Installs PostgreSQL 18 and the pgvector system extension on Debian/Ubuntu.
# Must be run once per machine before scripts/setup_db.sh.
#
# Usage:
#   sudo bash scripts/install_postgres.sh

set -euo pipefail

# ── PostgreSQL 18 ─────────────────────────────────────────────────────────────
echo "==> Adding PostgreSQL apt repository..."
apt-get install -y curl ca-certificates gnupg lsb-release

curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc \
    | gpg --dearmor -o /usr/share/keyrings/postgresql.gpg

echo "deb [signed-by=/usr/share/keyrings/postgresql.gpg] \
https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" \
    > /etc/apt/sources.list.d/pgdg.list

apt-get update
apt-get install -y postgresql-18

systemctl enable --now postgresql

# ── pgvector ──────────────────────────────────────────────────────────────────
# pgvector must be installed as a PostgreSQL system extension.
# It is NOT available via pip and cannot be installed from Python.
echo "==> Installing pgvector..."

if apt-get install -y postgresql-18-pgvector 2>/dev/null; then
    echo "    pgvector installed via apt."
else
    echo "    apt package not found — building from source..."
    apt-get install -y build-essential postgresql-server-dev-18 git
    git clone --depth 1 --branch v0.7.4 \
        https://github.com/pgvector/pgvector.git /tmp/pgvector
    make -C /tmp/pgvector
    make -C /tmp/pgvector install
    rm -rf /tmp/pgvector
    echo "    pgvector built and installed from source."
fi

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "==> Verifying pgvector is available to PostgreSQL..."
sudo -u postgres psql -tc \
    "SELECT name, default_version FROM pg_available_extensions WHERE name = 'vector';"

echo ""
echo "==> Done. Run scripts/setup_db.sh next to create the database."

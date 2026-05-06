#!/usr/bin/env bash
# scripts/entrypoint.sh
#
# Docker-only entrypoint. Not used for local development (run.py handles that).
# Responsibilities:
#   1. Load .env into the shell environment
#   2. Wait for PostgreSQL + pgvector to be ready
#   3. Exec Gunicorn (replaces the process — no wrapper process left running)

set -euo pipefail

# ── 1. Load .env ──────────────────────────────────────────────────────────────
# Python is already available in this image and handles quotes/comments/edge
# cases more reliably than a shell parser.  Only exports variables that are
# not already set in the environment (docker-compose env_file / -e flags win).
if [[ -f /app/.env ]]; then
    eval "$(python3 - <<'PYEOF'
import os

with open("/app/.env") as fh:
    for line in fh:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, raw = line.partition("=")
        key   = key.strip()
        value = raw.strip().strip("'\"")
        if key and key not in os.environ:
            # escape single-quotes so the eval is safe for any value
            safe = value.replace("'", "'\\''")
            print(f"export {key}='{safe}'")
PYEOF
)"
fi

# ── 2. Wait for PostgreSQL + pgvector ─────────────────────────────────────────
echo "==> Waiting for database..."
python3 - <<'PYEOF'
import os, sys, time
import psycopg2

dsn = (
    f"postgresql://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}"
    f"@{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}"
)

RETRIES = 10
DELAY   = 3.0

for attempt in range(1, RETRIES + 1):
    try:
        conn = psycopg2.connect(dsn, connect_timeout=3)
        cur  = conn.cursor()
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        if cur.fetchone() is None:
            print(
                "ERROR: PostgreSQL is reachable but the 'vector' extension is "
                "not installed.\n"
                "       Ensure scripts/init_db.sh ran inside the db container.",
                file=sys.stderr,
            )
            sys.exit(1)
        cur.close()
        conn.close()
        print("==> Database ready.")
        sys.exit(0)
    except Exception as exc:
        print(f"    Attempt {attempt}/{RETRIES}: {exc}", file=sys.stderr)
        if attempt < RETRIES:
            time.sleep(DELAY)

print(
    f"ERROR: Database not available after {RETRIES} attempts "
    f"({RETRIES * DELAY:.0f}s).\n"
    "       Check that PostgreSQL is running and DB_* variables are correct.",
    file=sys.stderr,
)
sys.exit(1)
PYEOF

# ── 3. Start Gunicorn ─────────────────────────────────────────────────────────
#
# -w 1          REQUIRED. YOLO, InsightFace, and StreamManager are in-process
#               GPU singletons. Multiple workers would each load the model
#               independently (doubling GPU memory) and stream state would be
#               siloed across workers with no shared memory. Scale horizontally
#               with multiple containers instead.
#
# --timeout 180 InsightFace (buffalo_l) may download on first run and can take
#               several minutes. 180 s is the worker boot budget; in-flight
#               request timeout is separate.
#
# No --preload  create_app() starts background threads (TaskQueue, YOLOBatcher,
#               StreamManager). Forking after thread creation is unsafe — threads
#               do not survive fork into worker processes. Each worker must call
#               create_app() itself.
# --threads 4   : allow up to 4 concurrent requests within the single worker
#                 process. Required for browser_webcam frame POST requests
#                 (15/s) to be handled alongside polling endpoints without
#                 queuing up behind each other.
#
# Logs to stdout/stderr so `docker logs` captures everything.

echo "==> Starting Gunicorn (1 worker)..."
exec gunicorn "app:create_app()" \
    --workers       1 \
    --threads       4 \
    --bind          0.0.0.0:5000 \
    --timeout       180 \
    --access-logfile  - \
    --error-logfile   -
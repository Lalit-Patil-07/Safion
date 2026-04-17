"""
Entry point for development.
For production use gunicorn:
    gunicorn "app:create_app()" -w 1 -b 0.0.0.0:5000 --timeout 120
Note: -w 1 (single worker) is intentional — the YOLO model and face pipeline
are in-process singletons.  Scale horizontally with multiple containers,
not multiple gunicorn workers in the same process.
"""
import os
import subprocess
import sys


def _db_ready() -> bool:
    """
    Return True if PostgreSQL is reachable AND the vector extension is installed.
    A single fast query — adds no measurable latency on normal starts.
    """
    try:
        import psycopg2
        from config import Config
        conn = psycopg2.connect(Config.SQLALCHEMY_DATABASE_URI, connect_timeout=3)
        cur  = conn.cursor()
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        result = cur.fetchone()
        cur.close()
        conn.close()
        return result is not None
    except Exception:
        return False


def _run_setup() -> None:
    """
    Locate and run scripts/setup_db.sh.
    Exits the process on failure so the app never starts against a broken DB.
    """
    script = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "setup_db.sh")
    )
    if not os.path.exists(script):
        print(f"ERROR: setup script not found at {script}", file=sys.stderr)
        print("       Run scripts/setup_db.sh manually — see README.", file=sys.stderr)
        sys.exit(1)

    print("==> Database not ready. Running scripts/setup_db.sh ...")
    result = subprocess.run(["bash", script])
    if result.returncode != 0:
        print("ERROR: Database setup failed. See output above.", file=sys.stderr)
        sys.exit(1)
    print("==> Database setup complete.\n")


if __name__ == "__main__":
    # Auto-setup: only runs if the DB is unreachable or vector extension is missing.
    # On every normal start this is a single connection + one query (< 5 ms).
    if not _db_ready():
        _run_setup()
        # Re-check: if setup ran but DB is still not ready, fail clearly.
        if not _db_ready():
            print(
                "ERROR: Database still not ready after setup.\n"
                "       Check PostgreSQL is running and pgvector is installed — see README.",
                file=sys.stderr,
            )
            sys.exit(1)

    from app import create_app
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
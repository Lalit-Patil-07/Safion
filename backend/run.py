"""
Entry point for development and production.
For production use gunicorn:
    gunicorn "app:create_app()" -w 1 -b 0.0.0.0:5000 --timeout 120
Note: -w 1 (single worker) is intentional — the YOLO model and face pipeline
are in-process singletons. Scale horizontally with multiple containers,
not multiple gunicorn workers in the same process.
"""
import os
import sys
import time


def _load_dotenv() -> None:
    """
    Read .env from the project root and inject any missing variables into
    os.environ. Does not override variables already set in the shell.
    Supports: KEY=value, KEY="value", KEY='value', inline # comments.
    No external dependencies.
    """
    env_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".env")
    )
    if not os.path.isfile(env_path):
        return

    with open(env_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, raw_value = line.partition("=")
            key   = key.strip()
            value = raw_value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


def _wait_for_db(retries: int = 10, delay: float = 3.0) -> None:
    """
    Block until PostgreSQL is reachable and the vector extension is present,
    or exit with a clear error after `retries` attempts.

    Uses only psycopg2 (already a hard dependency) and the same DSN the
    application uses — no extra dependencies, no schema changes.
    """
    import psycopg2
    from config import Config

    dsn = Config.SQLALCHEMY_DATABASE_URI

    for attempt in range(1, retries + 1):
        try:
            conn = psycopg2.connect(dsn, connect_timeout=3)
            cur  = conn.cursor()
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            result = cur.fetchone()
            cur.close()
            conn.close()

            if result is None:
                # DB is up but vector extension is missing — surface clearly.
                print(
                    "ERROR: PostgreSQL is reachable but the 'vector' extension "
                    "is not installed.\n"
                    "       Ensure init_db.sh ran inside the db container, or "
                    "run: CREATE EXTENSION IF NOT EXISTS vector;",
                    file=sys.stderr,
                )
                sys.exit(1)

            return  # ready

        except Exception as exc:
            print(
                f"==> Waiting for database... (attempt {attempt}/{retries}): {exc}",
                file=sys.stderr,
            )
            if attempt < retries:
                time.sleep(delay)

    print(
        f"ERROR: Database not available after {retries} attempts ({retries * delay:.0f}s).\n"
        "       Check that PostgreSQL is running and DB_* variables are correct.",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    _load_dotenv()
    _wait_for_db(retries=10, delay=3.0)

    from app import create_app
    from extensions import db
    from auth.utils import ensure_admin_user

    app = create_app(config_overrides={"_SKIP_SERVICES": True})

    with app.app_context():
        db.create_all()

    ensure_admin_user(app)

    from app import _init_services
    _init_services(app)

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
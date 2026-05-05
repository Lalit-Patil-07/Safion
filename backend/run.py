"""
Local development entry point.
-------------------------------
NOT used in Docker — the container uses scripts/entrypoint.sh + Gunicorn.

For production use Gunicorn directly:
    gunicorn "app:create_app()" -w 1 -b 0.0.0.0:5000 --timeout 180

Note: -w 1 (single worker) is intentional — YOLO, InsightFace, and
StreamManager are in-process GPU singletons.  Scale horizontally with
multiple containers, not multiple gunicorn workers in the same process.
"""
import os
import sys
import time


def _load_dotenv() -> None:
    """
    Read .env from the project root and inject any missing variables into
    os.environ.  Does not override variables already set in the shell.
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
                print(
                    "ERROR: PostgreSQL is reachable but the 'vector' extension "
                    "is not installed.\n"
                    "       Run: CREATE EXTENSION IF NOT EXISTS vector;",
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
        f"ERROR: Database not available after {retries} attempts "
        f"({retries * delay:.0f}s).\n"
        "       Check that PostgreSQL is running and DB_* variables are correct.",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    _load_dotenv()
    _wait_for_db(retries=10, delay=3.0)

    # create_app() handles db.create_all(), admin bootstrap, and service init.
    from app import create_app
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
import os
import sys
from alembic import context
from sqlalchemy import create_engine, pool, text

_here = os.path.dirname(os.path.abspath(__file__))
_backend = os.path.dirname(_here)
sys.path.insert(0, _backend)

from extensions import db
import auth.models
import face.models

target_metadata = db.metadata

def _build_db_url():
    return (
        f"postgresql://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}"
        f"@{os.environ['DB_HOST']}:{os.environ.get('DB_PORT','5432')}"
        f"/{os.environ['DB_NAME']}"
    )

def run_migrations_online():
    connectable = create_engine(_build_db_url(), poolclass=pool.NullPool)
    with connectable.connect() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        connection.commit()
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    context.configure(url=_build_db_url(), target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()
else:
    run_migrations_online()
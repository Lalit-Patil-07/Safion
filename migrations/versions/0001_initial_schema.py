from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = "0001"
down_revision = None

def upgrade():
    op.create_table(
        "users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("username", sa.String(80), nullable=False),
        sa.Column("email", sa.String(254)),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_users_username", "users", ["username"], unique=True)
    op.create_index("ix_users_email", "users", ["email"], unique=True)
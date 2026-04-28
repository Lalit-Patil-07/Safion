from alembic import command
from alembic.config import Config

def upgrade():
    command.upgrade(Config("alembic.ini"), "head")

if __name__ == "__main__":
    upgrade()
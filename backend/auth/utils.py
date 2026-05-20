"""Authentication utilities — password hashing and admin bootstrapping."""
import logging
import os

from extensions import bcrypt

logger = logging.getLogger(__name__)


def hash_password(password: str) -> str:
    return bcrypt.generate_password_hash(password).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.check_password_hash(password_hash, password)


def ensure_admin_user(app) -> None:
    """
    Idempotent startup guard — creates a default admin if none exists.

    Reads from environment:
        DEFAULT_ADMIN_USERNAME  (required)
        DEFAULT_ADMIN_PASSWORD  (required)
        DEFAULT_ADMIN_EMAIL     (optional)

    Behaviour:
        - Admin exists          → log and return, no writes
        - Credentials missing   → log warning and return, no crash
        - No admin exists       → create one and log confirmation
    """
    from auth.models import User
    from extensions import db

    with app.app_context():
        username = os.environ.get("DEFAULT_ADMIN_USERNAME", "").strip()
        password = os.environ.get("DEFAULT_ADMIN_PASSWORD", "").strip()
        email    = os.environ.get("DEFAULT_ADMIN_EMAIL",    "").strip().lower() or None

        if User.query.filter_by(role="admin").first():
            logger.info("Admin user already exists — skipping default admin creation.")
            return

        if not username or not password:
            logger.warning(
                "Admin credentials not provided — "
                "set DEFAULT_ADMIN_USERNAME and DEFAULT_ADMIN_PASSWORD "
                "to create a default admin on startup."
            )
            return

        if User.query.filter_by(username=username).first():
            logger.warning(
                "DEFAULT_ADMIN_USERNAME '%s' already exists as a non-admin user — "
                "skipping default admin creation.",
                username,
            )
            return

        if email and User.query.filter_by(email=email).first():
            logger.warning(
                "DEFAULT_ADMIN_EMAIL '%s' already in use — "
                "skipping default admin creation.",
                email,
            )
            return

        user = User(
            username=username,
            email=email,
            password_hash=hash_password(password),
            role="admin",
        )
        db.session.add(user)
        db.session.commit()
        logger.info("Admin user created: username='%s'.", username)
from extensions import bcrypt


def hash_password(password: str) -> str:
    return bcrypt.generate_password_hash(password).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.check_password_hash(password_hash, password)
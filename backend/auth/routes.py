from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import (
    create_access_token,
    create_refresh_token,
    jwt_required,
    get_jwt_identity,
    get_jwt,
)
from extensions import db, bcrypt
from auth.models import User

auth_bp = Blueprint("auth", __name__, url_prefix="/api/v1/auth")


def _validate_registration_payload(data: dict) -> str | None:
    """Return an error string if the payload is invalid, else None."""
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not username:
        return "Username is required."
    if len(username) < 3 or len(username) > 80:
        return "Username must be between 3 and 80 characters."
    if not password:
        return "Password is required."
    if len(password) < 8:
        return "Password must be at least 8 characters."
    return None


@auth_bp.post("/register")
def register():
    data = request.get_json(silent=True) or {}

    error = _validate_registration_payload(data)
    if error:
        return jsonify({"error": error}), 400

    username = data["username"].strip()

    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already taken."}), 409

    role = "operator"
    # First ever user becomes admin automatically
    if User.query.count() == 0:
        role = "admin"

    password_hash = bcrypt.generate_password_hash(data["password"]).decode("utf-8")
    user = User(username=username, password_hash=password_hash, role=role)
    db.session.add(user)
    db.session.commit()

    return jsonify({"message": "User created.", "user": user.to_dict()}), 201


@auth_bp.post("/login")
def login():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not username or not password:
        return jsonify({"error": "Username and password are required."}), 400

    user: User | None = User.query.filter_by(username=username, is_active=True).first()

    if not user or not bcrypt.check_password_hash(user.password_hash, password):
        # Deliberately vague — don't reveal whether the username exists
        return jsonify({"error": "Invalid credentials."}), 401

    additional_claims = {"role": user.role, "username": user.username}
    access_token = create_access_token(
        identity=user.id, additional_claims=additional_claims
    )
    refresh_token = create_refresh_token(
        identity=user.id, additional_claims=additional_claims
    )

    return jsonify({
        "access_token": access_token,
        "refresh_token": refresh_token,
        "user": user.to_dict(),
    }), 200


@auth_bp.post("/refresh")
@jwt_required(refresh=True)
def refresh():
    identity = get_jwt_identity()
    claims = get_jwt()

    user: User | None = User.query.filter_by(id=identity, is_active=True).first()
    if not user:
        return jsonify({"error": "User not found or deactivated."}), 401

    additional_claims = {"role": user.role, "username": user.username}
    access_token = create_access_token(
        identity=identity, additional_claims=additional_claims
    )
    return jsonify({"access_token": access_token}), 200


@auth_bp.get("/me")
@jwt_required()
def me():
    user_id = get_jwt_identity()
    user: User | None = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found."}), 404
    return jsonify(user.to_dict()), 200

"""
Authentication API endpoints.

Prefix: /api/v1/auth

Provides user registration, login (JWT access + refresh tokens),
token refresh, and current-user lookup.
"""
import re

from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import (
    create_access_token,
    create_refresh_token,
    jwt_required,
    get_jwt_identity,
    get_jwt,
    set_access_cookies,
    set_refresh_cookies,
    unset_jwt_cookies,
    get_csrf_token,
)
from extensions import db, limiter
from auth.models import User
from auth.utils import hash_password, verify_password

auth_bp = Blueprint("auth", __name__, url_prefix="/api/v1/auth")


def _validate_registration_payload(data: dict) -> str | None:
    """Return an error string if the payload is invalid, else None."""
    username = (data.get("username") or "").strip()
    email    = (data.get("email")    or "").strip().lower()
    password = data.get("password") or ""

    if not username:
        return "Username is required."
    if len(username) < 3 or len(username) > 80:
        return "Username must be between 3 and 80 characters."
    if email and not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return "Invalid email address."
    if not password:
        return "Password is required."
    if len(password) < 8:
        return "Password must be at least 8 characters."
    return None


@auth_bp.post("/register")
@limiter.limit("3/hour")
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

    email = (data.get("email") or "").strip().lower() or None
    if email and User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered."}), 409
    user = User(
        username=username,
        email=email,
        password_hash=hash_password(data["password"]),
        role=role,
    )
    db.session.add(user)
    db.session.commit()

    return jsonify({"message": "User created.", "user": user.to_dict()}), 201


@auth_bp.post("/login")
@limiter.limit("5/minute")
def login():
    data = request.get_json(silent=True) or {}
    identifier = (data.get("email") or data.get("username") or "").strip()
    password   =  data.get("password") or ""

    if not identifier or not password:
        return jsonify({"error": "Credentials are required."}), 400

    user: User | None = (
        User.query
        .filter_by(is_active=True)
        .filter(
            (User.email    == identifier.lower()) |
            (User.username == identifier)
        )
        .first()
    )

    if not user or not verify_password(password, user.password_hash):
        return jsonify({"error": "Invalid credentials."}), 401

    additional_claims = {"role": user.role, "username": user.username}
    access_token = create_access_token(
        identity=user.id, additional_claims=additional_claims
    )
    refresh_token = create_refresh_token(
        identity=user.id, additional_claims=additional_claims
    )

    response = jsonify({
        "user": user.to_dict(),
        "csrf_token": get_csrf_token(access_token),
    })
    set_access_cookies(response, access_token)
    set_refresh_cookies(response, refresh_token)
    return response, 200


@auth_bp.post("/refresh")
@limiter.limit("10/minute")
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
    response = jsonify({"csrf_token": get_csrf_token(access_token)})
    set_access_cookies(response, access_token)
    return response, 200


@auth_bp.post("/logout")
def logout():
    response = jsonify({"message": "Logged out."})
    unset_jwt_cookies(response)
    return response, 200


@auth_bp.get("/me")
@jwt_required()
def me():
    user_id = get_jwt_identity()
    user: User | None = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found."}), 404
    return jsonify(user.to_dict()), 200

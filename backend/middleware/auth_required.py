"""
Reusable auth decorators.

Usage:
    from middleware.auth_required import jwt_required_route, admin_required

    @app.route("/protected")
    @jwt_required_route()
    def protected():
        ...

    @app.route("/admin-only")
    @admin_required()
    def admin_only():
        ...
"""
from functools import wraps
from flask import jsonify
from flask_jwt_extended import jwt_required, get_jwt, verify_jwt_in_request


def jwt_required_route(optional: bool = False):
    """
    Standard JWT guard.  Wraps flask_jwt_extended's jwt_required so our routes
    don't need to import flask_jwt_extended directly.
    """
    def decorator(fn):
        @wraps(fn)
        @jwt_required(optional=optional)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def admin_required():
    """
    Requires a valid JWT **and** role == 'admin'.
    Returns 403 for authenticated non-admins.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                verify_jwt_in_request()
            except Exception:
                return jsonify({"error": "Authentication required."}), 401

            claims = get_jwt()
            if claims.get("role") != "admin":
                return jsonify({"error": "Admin privileges required."}), 403

            return fn(*args, **kwargs)
        return wrapper
    return decorator

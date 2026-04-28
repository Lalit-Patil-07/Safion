"""
Auth decorators and blueprint-level guards.

Usage:
    from middleware.auth_required import require_auth, require_admin, protect_blueprint
"""
from functools import wraps
from flask import jsonify, request
from flask_jwt_extended import jwt_required, get_jwt, verify_jwt_in_request


def jwt_required_route(optional: bool = False):
    """Standard JWT guard."""
    def decorator(fn):
        @wraps(fn)
        @jwt_required(optional=optional)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def admin_required():
    """Requires valid JWT and role == 'admin'."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                verify_jwt_in_request()
            except Exception:
                return jsonify({"error": "Authentication required."}), 401
            if get_jwt().get("role") != "admin":
                return jsonify({"error": "Admin privileges required."}), 403
            return fn(*args, **kwargs)
        return wrapper
    return decorator


# Canonical names used across the codebase
require_auth  = jwt_required_route
require_admin = admin_required


def protect_blueprint(bp, *, exempt_endpoints: set | None = None) -> None:
    """
    Attach a JWT before_request guard to an entire blueprint.

    Parameters
    ----------
    bp               : Flask Blueprint
    exempt_endpoints : fully-qualified endpoint names to skip,
                       e.g. {"streams.video_feed"}
    """
    _exempt = frozenset(exempt_endpoints or ())

    @bp.before_request
    def _guard():
        if request.endpoint in _exempt:
            return
        try:
            verify_jwt_in_request()
        except Exception:
            return jsonify({"error": "Authentication required."}), 401

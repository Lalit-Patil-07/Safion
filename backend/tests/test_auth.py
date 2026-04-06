"""
Tests for the auth blueprint.
Run with: pytest tests/test_auth.py -v
"""
import pytest
from app import create_app
from extensions import db as _db
from config import Config


class TestConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    JWT_SECRET_KEY = "test-jwt-secret"
    SECRET_KEY = "test-secret"
    # Disable service init for unit tests
    _SKIP_SERVICES = True


@pytest.fixture()
def app():
    app = create_app(TestConfig)
    with app.app_context():
        _db.create_all()
        yield app
        _db.session.remove()
        _db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
class TestRegister:
    def test_first_user_becomes_admin(self, client):
        r = client.post("/api/v1/auth/register", json={
            "username": "alice", "password": "securepass"
        })
        assert r.status_code == 201
        assert r.json["user"]["role"] == "admin"

    def test_second_user_is_operator(self, client):
        client.post("/api/v1/auth/register", json={"username": "alice", "password": "pass1234"})
        r = client.post("/api/v1/auth/register", json={"username": "bob", "password": "pass1234"})
        assert r.status_code == 201
        assert r.json["user"]["role"] == "operator"

    def test_duplicate_username_rejected(self, client):
        client.post("/api/v1/auth/register", json={"username": "alice", "password": "pass1234"})
        r = client.post("/api/v1/auth/register", json={"username": "alice", "password": "pass1234"})
        assert r.status_code == 409

    def test_short_password_rejected(self, client):
        r = client.post("/api/v1/auth/register", json={"username": "alice", "password": "short"})
        assert r.status_code == 400

    def test_missing_username_rejected(self, client):
        r = client.post("/api/v1/auth/register", json={"password": "pass1234"})
        assert r.status_code == 400

    def test_short_username_rejected(self, client):
        r = client.post("/api/v1/auth/register", json={"username": "ab", "password": "pass1234"})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------
class TestLogin:
    def _register(self, client, username="alice", password="pass1234"):
        client.post("/api/v1/auth/register", json={"username": username, "password": password})

    def test_valid_login_returns_tokens(self, client):
        self._register(client)
        r = client.post("/api/v1/auth/login", json={"username": "alice", "password": "pass1234"})
        assert r.status_code == 200
        assert "access_token" in r.json
        assert "refresh_token" in r.json

    def test_wrong_password_rejected(self, client):
        self._register(client)
        r = client.post("/api/v1/auth/login", json={"username": "alice", "password": "wrong"})
        assert r.status_code == 401

    def test_unknown_user_rejected(self, client):
        r = client.post("/api/v1/auth/login", json={"username": "nobody", "password": "pass1234"})
        assert r.status_code == 401

    def test_missing_credentials_rejected(self, client):
        r = client.post("/api/v1/auth/login", json={})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Protected routes
# ---------------------------------------------------------------------------
class TestProtectedRoutes:
    def _get_token(self, client):
        client.post("/api/v1/auth/register", json={"username": "alice", "password": "pass1234"})
        r = client.post("/api/v1/auth/login", json={"username": "alice", "password": "pass1234"})
        return r.json["access_token"]

    def test_me_requires_auth(self, client):
        r = client.get("/api/v1/auth/me")
        assert r.status_code == 401

    def test_me_returns_user_info(self, client):
        token = self._get_token(client)
        r = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert r.json["username"] == "alice"

    def test_refresh_returns_new_access_token(self, client):
        client.post("/api/v1/auth/register", json={"username": "alice", "password": "pass1234"})
        login = client.post("/api/v1/auth/login", json={"username": "alice", "password": "pass1234"})
        refresh_token = login.json["refresh_token"]
        r = client.post(
            "/api/v1/auth/refresh",
            headers={"Authorization": f"Bearer {refresh_token}"},
        )
        assert r.status_code == 200
        assert "access_token" in r.json

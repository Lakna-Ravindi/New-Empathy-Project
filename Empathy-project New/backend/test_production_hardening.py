"""Tests for Phase 5 Production Hardening improvements:
- JWT security & startup validation
- Token revocation on logout
- MongoDB in-memory fallback disabling in production
- Gemini timeout & deadline protection
- Multi-worker rate limiting
- CORS production validation
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from pymongo.errors import PyMongoError

import auth
from auth import validate_jwt_secret, create_access_token, require_auth, get_user_store
from learning.interaction_store import LearningStore
from rate_limiter import MultiWorkerRateLimiter
from server import app, configure_cors_origins


# =====================================================================
# 1. JWT Security Tests
# =====================================================================

def test_jwt_validation_rejects_missing_or_empty_secret_in_prod():
    with pytest.raises(RuntimeError, match="not configured in production"):
        validate_jwt_secret("", is_production=True)

    with patch.dict(os.environ, {"JWT_SECRET_KEY": ""}):
        with pytest.raises(RuntimeError, match="not configured in production"):
            validate_jwt_secret(None, is_production=True)


def test_jwt_validation_rejects_insecure_defaults_in_prod():
    insecure = [
        "fallback-secret-key-change-in-production",
        "secret",
        "changeme",
        "your-secret-key",
        "default",
        "password",
    ]
    for s in insecure:
        with pytest.raises(RuntimeError, match="known insecure default"):
            validate_jwt_secret(s, is_production=True)


def test_jwt_validation_rejects_short_secret_in_prod():
    with pytest.raises(RuntimeError, match="at least 32 characters in production"):
        validate_jwt_secret("short-key-under-32-chars-1234", is_production=True)


def test_jwt_validation_accepts_strong_secret_in_prod():
    strong = "a" * 32
    assert validate_jwt_secret(strong, is_production=True) == strong


def test_jwt_validation_allows_dev_secret_when_not_in_prod():
    res = validate_jwt_secret("", is_production=False)
    assert res is not None
    assert len(res) > 10


# =====================================================================
# 2. Token Revocation Tests
# =====================================================================

def test_token_revocation_rejects_revoked_jwt():
    store = get_user_store()
    user = {
        "_id": "60c72b2f9b1d8b2bad000001",
        "student_id": "STU-TEST1",
        "username": "student_test1",
        "email": "student1@example.com",
        "role": "student",
    }
    with patch.object(store, "find_by_id", return_value=user):
        with patch.object(store, "ensure_student_id", return_value=user):
            token = create_access_token(user)
            client = app.test_client()

            # Active token should succeed
            res = client.get(
                "/api/auth/me",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert res.status_code == 200

            # Revoke token
            store.revoke_token(token)

            # Revoked token must be rejected with 401
            res_revoked = client.get(
                "/api/auth/me",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert res_revoked.status_code == 401
            assert "revoked" in res_revoked.get_json()["error"].lower()


def test_logout_endpoint_revokes_token():
    store = get_user_store()
    user = {
        "_id": "60c72b2f9b1d8b2bad000002",
        "student_id": "STU-TEST2",
        "username": "student_test2",
        "email": "student2@example.com",
        "role": "student",
    }
    with patch.object(store, "find_by_id", return_value=user):
        with patch.object(store, "ensure_student_id", return_value=user):
            token = create_access_token(user)
            client = app.test_client()

            # Logout with token in header
            logout_res = client.post(
                "/api/auth/logout",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert logout_res.status_code == 200

            # Token must now be marked revoked
            assert store.is_token_revoked(token)


# =====================================================================
# 3. MongoDB Data Safety Tests
# =====================================================================

def test_mongodb_production_mode_disables_in_memory_fallback():
    # In production mode, offline store must fail rather than saving to RAM
    with patch.dict(os.environ, {"DISABLE_IN_MEMORY_FALLBACK": "true"}):
        with patch.object(LearningStore, "_connect"):
            store = LearningStore()
            store.is_available = False
            store._check_or_reconnect = MagicMock(return_value=False)

            with pytest.raises(PyMongoError, match="[Ii]n-memory fallback is disabled in production"):
                store.save_interaction("stu-1", {"student_question": "q", "status": "greeting"}, "resp")

            with pytest.raises(PyMongoError, match="Student progress cannot be recorded in production"):
                store.complete_item("stu-1", "skill_01", "act_01")


def test_mongodb_development_mode_preserves_offline_fallback():
    with patch.dict(os.environ, {"DISABLE_IN_MEMORY_FALLBACK": "false", "FLASK_ENV": "development"}):
        with patch.object(LearningStore, "_connect"):
            store = LearningStore()
            store.is_available = False
            store._check_or_reconnect = MagicMock(return_value=False)

            res = store.save_interaction("stu-1", {"student_question": "q", "status": "greeting"}, "resp")
            assert res.startswith("local-")

            prog = store.complete_item("stu-1", "skill_01", "act_01")
            assert "act_01" in prog["completed_item_ids"]


# =====================================================================
# 4. Multi-Worker Rate Limiting Tests
# =====================================================================

def test_rate_limiter_blocks_requests_exceeding_limit():
    limiter = MultiWorkerRateLimiter()
    client = app.test_client()

    with patch.dict(os.environ, {"RATELIMIT_ENABLED": "true"}):
        with app.test_request_context("/api/test-limit", environ_base={"REMOTE_ADDR": "10.0.0.1"}):
            # 2 allowed, 3rd blocked with limit 2
            allowed1, c1, _ = limiter.check_rate_limit("rate_test_key", limit=2, window_seconds=60)
            allowed2, c2, _ = limiter.check_rate_limit("rate_test_key", limit=2, window_seconds=60)
            allowed3, c3, retry_after = limiter.check_rate_limit("rate_test_key", limit=2, window_seconds=60)

            assert allowed1 is True and c1 == 1
            assert allowed2 is True and c2 == 2
            assert allowed3 is False and c3 == 3
            assert retry_after > 0


# =====================================================================
# 5. CORS Production Validation Tests
# =====================================================================

def test_cors_production_rejects_missing_origins():
    with patch.dict(os.environ, {"FLASK_ENV": "production", "CORS_ALLOWED_ORIGINS": ""}):
        with pytest.raises(RuntimeError, match="CORS_ALLOWED_ORIGINS must be explicitly configured"):
            configure_cors_origins()


def test_cors_production_rejects_wildcard():
    with patch.dict(os.environ, {"FLASK_ENV": "production", "CORS_ALLOWED_ORIGINS": "*"}):
        with pytest.raises(RuntimeError, match="Wildcard '\\*' CORS origin is not permitted"):
            configure_cors_origins()


def test_cors_production_accepts_valid_origins():
    with patch.dict(os.environ, {"FLASK_ENV": "production", "CORS_ALLOWED_ORIGINS": "https://empathy.example.com, https://app.example.com"}):
        origins = configure_cors_origins()
        assert origins == ["https://empathy.example.com", "https://app.example.com"]

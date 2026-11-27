"""Flask integration verification for ObjectiveCompletionAdapter.

Tests the adapter through Flask's request context, JSON parsing,
JWT authentication via @require_auth, and response generation,
verifying all 6 required HTTP status codes:
1. Authenticated successful completion – HTTP 200.
2. Invalid skill/objective – HTTP 400.
3. Missing or invalid authentication – HTTP 401.
4. Client identity mismatch – HTTP 403.
5. Missing prerequisites – HTTP 422.
6. Database failure – HTTP 503.
"""

from unittest.mock import MagicMock, patch
import jwt
import pytest
from flask import Flask, jsonify, request, g
from pymongo.errors import PyMongoError

import auth
import server
from agents.completion_adapter import ObjectiveCompletionAdapter
from agents.tools.curriculum_tools import get_progress_structure
from learning.progress_service import ProgressService


class UserStoreStub:
    def find_by_id(self, user_id):
        return {
            "_id": user_id,
            "username": "student",
            "email": "student@example.test",
            "role": "student",
            "student_id": user_id,
        }

    def ensure_student_id(self, user):
        return user


@pytest.fixture
def auth_setup():
    """Configures JWT and user store for Flask test client."""
    stub = UserStoreStub()
    with patch.object(auth, "JWT_SECRET_KEY", "integration-test-secret"), \
         patch.object(auth, "get_user_store", return_value=stub):
        yield stub


def _make_token(student_id="student-1"):
    return jwt.encode(
        {"sub": student_id, "student_id": student_id, "exp": 4_102_444_800},
        "integration-test-secret",
        algorithm="HS256",
    )


@pytest.fixture
def flask_test_app(auth_setup):
    """Creates a dedicated Flask test application mounting the exact cutover endpoint structure."""
    test_app = Flask("test_integration_app")

    mock_store = MagicMock(spec=["get_progress", "get_interaction", "complete_objective"])
    mock_service = ProgressService(structure=get_progress_structure())

    mock_store.get_progress.return_value = {
        "student_id": "student-1",
        "skill_id": "skill_01",
        "completed_item_ids": [],
        "completed_objective_ids": [],
        "quiz_scores": {},
        "next_recommended_learning_objective": None,
        "updated_at": "2026-01-01T00:00:00Z",
    }
    mock_store.get_interaction.return_value = {
        "student_id": "student-1",
        "learning_context": {
            "skill": {"id": "skill_01"},
            "learning_objective": {"id": "obj_01_01"},
        },
        "learning_steps": [
            {
                "type": "practice_check",
                "question": "Is mindful breathing beneficial for calming?",
                "correct_answer": True,
            }
        ],
    }

    orchestrator = server.EmpathyOrchestrator(
        learning_store=mock_store,
        progress_service=mock_service,
    )

    # Exact proposed cutover endpoint implementation
    @test_app.post("/api/objectives/<objective_id>/complete")
    @auth.require_auth
    def test_complete_objective(objective_id):
        student_id = g.student_id
        body = request.get_json(silent=True) or {}
        payload, status = orchestrator.complete_objective_http(
            student_id=body.get("student_id") or student_id,
            authenticated_student_id=student_id,
            skill_id=body.get("skill_id"),
            objective_id=objective_id,
            body=body,
        )
        return jsonify(payload), status

    client = test_app.test_client()
    return client, mock_store, mock_service


# 1. Authenticated successful completion – HTTP 200
def test_flask_authenticated_successful_completion(flask_test_app):
    client, store, service = flask_test_app
    token = _make_token("student-1")

    progress_state = {
        "student_id": "student-1",
        "skill_id": "skill_01",
        "completed_item_ids": ["activity_01_01"],
        "completed_objective_ids": [],
        "quiz_scores": {},
        "next_recommended_learning_objective": None,
        "updated_at": "2026-01-01T00:00:00Z",
    }

    def mock_get(s_id, skill):
        return dict(progress_state)

    def mock_complete(s_id, skill, obj_rec, next_obj):
        progress_state["completed_objective_ids"] = ["obj_01_01"]
        progress_state["next_recommended_learning_objective"] = next_obj
        return dict(progress_state)

    store.get_progress.side_effect = mock_get
    store.complete_objective.side_effect = mock_complete

    res = client.post(
        "/api/objectives/obj_01_01/complete",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "skill_id": "skill_01",
            "source": "chatbot",
            "interaction_id": "inter-001",
            "quick_check_answer": True,
        },
    )

    assert res.status_code == 200
    data = res.get_json()
    assert data["student_id"] == "student-1"
    assert "obj_01_01" in data["completed_objective_ids"]
    assert data["next_recommended_learning_objective"]["id"] == "obj_01_05"


# 2. Invalid skill/objective – HTTP 400
def test_flask_invalid_skill_or_objective(flask_test_app):
    client, store, service = flask_test_app
    token = _make_token("student-1")

    # Invalid objective ID
    res = client.post(
        "/api/objectives/invalid_obj/complete",
        headers={"Authorization": f"Bearer {token}"},
        json={"skill_id": "skill_01"},
    )
    assert res.status_code == 400
    assert "valid skill_id and learning objective are required" in res.get_json()["error"]

    # Invalid skill ID
    res = client.post(
        "/api/objectives/obj_01_01/complete",
        headers={"Authorization": f"Bearer {token}"},
        json={"skill_id": "invalid_skill"},
    )
    assert res.status_code == 400
    assert "valid skill_id and learning objective are required" in res.get_json()["error"]

    # Missing JSON body
    res = client.post(
        "/api/objectives/obj_01_01/complete",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 400
    assert "valid skill_id and learning objective are required" in res.get_json()["error"]


# 3. Missing or invalid authentication – HTTP 401
def test_flask_missing_or_invalid_auth(flask_test_app):
    client, store, service = flask_test_app

    # Completely unauthenticated
    res = client.post(
        "/api/objectives/obj_01_01/complete",
        json={"skill_id": "skill_01"},
    )
    assert res.status_code == 401
    assert "Authentication is required." in res.get_json()["error"]

    # Malformed token
    res = client.post(
        "/api/objectives/obj_01_01/complete",
        headers={"Authorization": "Bearer bad-jwt-token"},
        json={"skill_id": "skill_01"},
    )
    assert res.status_code == 401
    assert "Invalid authentication token." in res.get_json()["error"]


# 4. Client identity mismatch – HTTP 403
def test_flask_client_identity_mismatch(flask_test_app):
    client, store, service = flask_test_app
    # Attacker authenticated as student-1
    token = _make_token("student-1")

    # Tries to mutate student-victim
    res = client.post(
        "/api/objectives/obj_01_01/complete",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "student_id": "student-victim",
            "skill_id": "skill_01",
        },
    )
    assert res.status_code == 403
    assert "Security violation" in res.get_json()["error"]


# 5. Missing prerequisites – HTTP 422
def test_flask_missing_prerequisites(flask_test_app):
    client, store, service = flask_test_app
    token = _make_token("student-1")

    # Missing activity
    store.get_progress.return_value = {
        "student_id": "student-1",
        "skill_id": "skill_01",
        "completed_item_ids": [],
        "completed_objective_ids": [],
        "quiz_scores": {},
    }

    res = client.post(
        "/api/objectives/obj_01_01/complete",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "skill_id": "skill_01",
            "source": "chatbot",
            "interaction_id": "inter-001",
            "quick_check_answer": True,
        },
    )

    assert res.status_code == 422
    data = res.get_json()
    assert data["completed"] is False
    assert data["objective_id"] == "obj_01_01"
    assert "activity" in data["missing_completion_conditions"]
    assert "Complete the objective activity before completing this objective." in data["error"]


# 6. Database failure – HTTP 503
def test_flask_database_failure(flask_test_app):
    client, store, service = flask_test_app
    token = _make_token("student-1")

    store.get_progress.side_effect = PyMongoError("MongoDB cluster unreachable.")

    res = client.post(
        "/api/objectives/obj_01_01/complete",
        headers={"Authorization": f"Bearer {token}"},
        json={"skill_id": "skill_01"},
    )

    assert res.status_code == 503
    assert res.get_json()["error"] == "Could not update student progress."

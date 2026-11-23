"""Phase 6B.1: Controlled Agent Completion Feature Flag Tests.

Validates that:
1. ENABLE_AGENT_OBJECTIVE_COMPLETION defaults to false.
2. When false (default), the legacy completion handler (_legacy_complete_objective) is called.
3. When true, orchestrator.complete_objective_http is called.
4. When true, g.student_id is strictly passed as authenticated_student_id.
5. Client-supplied student_id is preserved for adapter validation, preventing identity spoofing (HTTP 403).
6. Authenticated completion succeeds through the agent path when enabled (HTTP 200).
"""

from unittest.mock import MagicMock, patch
import jwt
import pytest
from pymongo.errors import PyMongoError

import auth
import server
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
def auth_client():
    stub = UserStoreStub()
    with patch.object(auth, "JWT_SECRET_KEY", "controlled-test-secret"), \
         patch.object(auth, "get_user_store", return_value=stub):
        token = jwt.encode(
            {"sub": "student-1", "student_id": "student-1", "exp": 4_102_444_800},
            "controlled-test-secret",
            algorithm="HS256",
        )
        test_client = server.app.test_client()
        test_client.environ_base["HTTP_AUTHORIZATION"] = f"Bearer {token}"
        yield test_client


def test_feature_flag_defaults_to_false(auth_client, monkeypatch):
    """When the environment variable is unset, the feature flag defaults to false and uses legacy handler."""
    monkeypatch.delenv("ENABLE_AGENT_OBJECTIVE_COMPLETION", raising=False)

    with patch.object(server, "_legacy_complete_objective", return_value=("legacy_called", 200)) as mock_legacy, \
         patch.object(server.orchestrator, "complete_objective_http") as mock_agent:

        res = auth_client.post(
            "/api/objectives/obj_01_01/complete",
            json={"skill_id": "skill_01"},
        )

        assert res.status_code == 200
        mock_legacy.assert_called_once()
        mock_agent.assert_not_called()


def test_feature_flag_explicit_false_calls_legacy(auth_client, monkeypatch):
    """When ENABLE_AGENT_OBJECTIVE_COMPLETION=false, legacy handler is invoked."""
    monkeypatch.setenv("ENABLE_AGENT_OBJECTIVE_COMPLETION", "false")

    with patch.object(server, "_legacy_complete_objective", return_value=("legacy_called", 200)) as mock_legacy, \
         patch.object(server.orchestrator, "complete_objective_http") as mock_agent:

        res = auth_client.post(
            "/api/objectives/obj_01_01/complete",
            json={"skill_id": "skill_01"},
        )

        assert res.status_code == 200
        mock_legacy.assert_called_once()
        mock_agent.assert_not_called()


def test_feature_flag_true_delegates_to_orchestrator(auth_client, monkeypatch):
    """When ENABLE_AGENT_OBJECTIVE_COMPLETION=true, orchestrator.complete_objective_http is invoked."""
    monkeypatch.setenv("ENABLE_AGENT_OBJECTIVE_COMPLETION", "true")

    dummy_response = {
        "student_id": "student-1",
        "skill_id": "skill_01",
        "completed_item_ids": ["activity_01_01"],
        "completed_objective_ids": ["obj_01_01"],
        "quiz_scores": {},
        "next_recommended_learning_objective": {"id": "obj_01_05"},
    }

    with patch.object(server, "_legacy_complete_objective") as mock_legacy, \
         patch.object(server.orchestrator, "complete_objective_http", return_value=(dummy_response, 200)) as mock_agent:

        body = {
            "skill_id": "skill_01",
            "source": "chatbot",
            "interaction_id": "inter-001",
            "quick_check_answer": True,
        }

        res = auth_client.post(
            "/api/objectives/obj_01_01/complete",
            json=body,
        )

        assert res.status_code == 200
        assert res.get_json()["completed_objective_ids"] == ["obj_01_01"]
        mock_legacy.assert_not_called()
        mock_agent.assert_called_once_with(
            student_id="student-1",
            authenticated_student_id="student-1",
            skill_id="skill_01",
            objective_id="obj_01_01",
            body=body,
        )


def test_feature_flag_true_preserves_client_student_id_for_adapter_validation(auth_client, monkeypatch):
    """When a client passes a different student_id in body, it is passed for adapter validation (rejecting with 403)."""
    monkeypatch.setenv("ENABLE_AGENT_OBJECTIVE_COMPLETION", "true")

    # Attacker authenticated as student-1, passes student-victim in body
    body = {
        "student_id": "student-victim",
        "skill_id": "skill_01",
    }

    res = auth_client.post(
        "/api/objectives/obj_01_01/complete",
        json=body,
    )

    assert res.status_code == 403
    assert "Security violation" in res.get_json()["error"]


def test_live_agent_completion_through_server_endpoint_when_enabled(auth_client, monkeypatch):
    """End-to-end integration test through server.py with live orchestrator when flag is enabled."""
    monkeypatch.setenv("ENABLE_AGENT_OBJECTIVE_COMPLETION", "true")

    mock_store = MagicMock(spec=["get_progress", "get_interaction", "complete_objective"])
    mock_service = ProgressService(structure=get_progress_structure())

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

    mock_store.get_progress.side_effect = mock_get
    mock_store.complete_objective.side_effect = mock_complete
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

    with patch.object(server.orchestrator, "_learning_store", mock_store), \
         patch.object(server.orchestrator, "_progress_service", mock_service), \
         patch.object(server, "learning_store", mock_store), \
         patch.object(server, "progress_service", mock_service):

        # Update the default bindings in progress_tools
        from agents.tools.progress_tools import set_default_learning_store, set_default_progress_service
        set_default_learning_store(mock_store)
        set_default_progress_service(mock_service)

        try:
            res = auth_client.post(
                "/api/objectives/obj_01_01/complete",
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
        finally:
            set_default_learning_store(None)
            set_default_progress_service(None)

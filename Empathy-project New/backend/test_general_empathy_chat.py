import sys
from pathlib import Path
from unittest.mock import Mock, patch

import jwt
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import auth
import server
from agents import (
    AgentExecutionError,
    GeneralEmpathyInput,
    GeneralEmpathyOutput,
)


class UserStoreStub:
    def find_by_id(self, user_id):
        return {
            "_id": user_id,
            "username": "student",
            "email": "student@example.test",
            "role": "student",
        }

    def ensure_student_id(self, user):
        return user


class SafetyStoreStub:
    """Mirrors the SafetyStore interface used by the chat endpoints."""

    def __init__(self, count=0, active=False, screening_required=False):
        self.count = count
        self.active = active
        self.screening_required = screening_required
        self.events = {}
        self.recorded = []

    def get_status(self, student_id):
        return {
            "student_id": student_id,
            "qualifying_event_count": self.count,
            "restriction_active": self.active,
            "screening_required": self.screening_required,
        }

    def record_event(self, student_id, message, risk, request_id):
        if request_id in self.events:
            return self.events[request_id], self.get_status(student_id), False
        self.count += 1
        self.active = self.count >= 3
        event = {
            "student_id": student_id,
            "message": message,
            "risk_level": risk["risk_level"],
            "immediate": risk["immediate"],
            "qualifying_event_count": self.count,
        }
        self.events[request_id] = event
        self.recorded.append(event)
        return event, self.get_status(student_id), True


@pytest.fixture
def safety_store():
    store = SafetyStoreStub()
    with patch.object(server, "safety_store", store):
        yield store


@pytest.fixture
def client():
    with patch.object(auth, "JWT_SECRET_KEY", "test-secret"), patch.object(
        auth, "get_user_store", return_value=UserStoreStub()
    ):
        token = jwt.encode(
            {"sub": "student-1", "exp": 4_102_444_800},
            "test-secret",
            algorithm="HS256",
        )
        test_client = server.app.test_client()
        test_client.environ_base["HTTP_AUTHORIZATION"] = f"Bearer {token}"
        yield test_client


def test_general_empathy_chat_requires_authentication():
    response = server.app.test_client().post(
        "/api/general-empathy-chat",
        json={"message": "I feel overwhelmed."},
    )

    assert response.status_code == 401


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": ""},
        {"message": "   "},
        {"message": 12},
        {"message": "I feel overwhelmed.", "history": "not-a-list"},
        {"message": "I feel overwhelmed.", "history": [{"role": "system", "content": "x"}]},
        {"message": "I feel overwhelmed.", "history": [{"role": [], "content": "x"}]},
        {"message": "I feel overwhelmed.", "history": [{"role": "user", "content": " "}]},
        {"message": "I" * 4001},
        {"message": "I feel overwhelmed.", "history": [{"role": "user", "content": "x" * 4001}]},
        {"message": "I feel overwhelmed.", "history": [{"role": "user", "content": "x"}] * 21},
    ],
)
def test_general_empathy_chat_rejects_invalid_input(client, payload):
    response = client.post("/api/general-empathy-chat", json=payload)

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_general_empathy_chat_calls_agent_without_learning_side_effects(client, safety_store):
    history = [
        {"role": "user", "content": "My friend has been distant."},
        {"role": "assistant", "content": "That sounds unsettling."},
    ]
    mock_output = GeneralEmpathyOutput(response="It makes sense that you feel hurt.")

    with patch.object(
        server.general_empathy_agent,
        "execute",
        return_value=mock_output,
    ) as execute_mock, patch.object(server.learning_store, "save_interaction") as save_mock:
        response = client.post(
            "/api/general-empathy-chat",
            json={"message": "I feel hurt.", "history": history, "request_id": "req-123"},
        )

    assert response.status_code == 200
    assert response.get_json() == {"response": "It makes sense that you feel hurt."}
    execute_mock.assert_called_once()
    call_arg = execute_mock.call_args[0][0]
    assert isinstance(call_arg, GeneralEmpathyInput)
    assert call_arg.student_id == "student-1"
    assert call_arg.authenticated_student_id == "student-1"
    assert call_arg.current_message == "I feel hurt."
    assert call_arg.conversation_history == history
    assert call_arg.request_id == "req-123"
    assert call_arg.fallback_on_error is False
    save_mock.assert_not_called()


def test_client_supplied_student_id_cannot_override_jwt_identity(client, safety_store):
    mock_output = GeneralEmpathyOutput(response="I am here for you.")

    with patch.object(
        server.general_empathy_agent,
        "execute",
        return_value=mock_output,
    ) as execute_mock:
        response = client.post(
            "/api/general-empathy-chat",
            json={"message": "I am stressed.", "student_id": "malicious-student-id"},
        )

    assert response.status_code == 200
    call_arg = execute_mock.call_args[0][0]
    # Sourced strictly from g.student_id (student-1 from JWT)
    assert call_arg.student_id == "student-1"
    assert call_arg.authenticated_student_id == "student-1"


def test_general_empathy_chat_returns_unavailable_on_agent_failure(client, safety_store):
    with patch.object(
        server.general_empathy_agent,
        "execute",
        side_effect=AgentExecutionError("Gemini service temporarily unavailable."),
    ):
        response = client.post(
            "/api/general-empathy-chat",
            json={"message": "I need someone to listen."},
        )

    assert response.status_code == 503
    assert "unavailable" in response.get_json()["error"]


# ---------------------------------------------------
# Safety screening on general empathy chat
# ---------------------------------------------------
def test_normal_message_passes_safety_and_reaches_agent(client, safety_store):
    mock_output = GeneralEmpathyOutput(response="I hear you.")
    with patch.object(
        server.general_empathy_agent,
        "execute",
        return_value=mock_output,
    ) as execute_mock:
        response = client.post(
            "/api/general-empathy-chat",
            json={"message": "I am stressed about exams."},
        )

    assert response.status_code == 200
    assert response.get_json() == {"response": "I hear you."}
    execute_mock.assert_called_once()
    assert safety_store.recorded == []
    assert safety_store.count == 0


@pytest.mark.parametrize(
    "message",
    [
        "I want to die.",
        "I want to kill myself.",
        "I feel completely hopeless.",
    ],
)
def test_red_flag_message_returns_safety_response_without_agent(client, safety_store, message):
    with patch.object(server.general_empathy_agent, "execute") as execute_mock:
        response = client.post(
            "/api/general-empathy-chat",
            json={"message": message, "request_id": "flag-1"},
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["safety"] is True
    assert payload["qualifying_event_count"] == 1
    assert "response" not in payload
    execute_mock.assert_not_called()
    assert len(safety_store.recorded) == 1
    assert safety_store.recorded[0]["student_id"] == "student-1"
    assert safety_store.recorded[0]["message"] == message


def test_immediate_risk_message_returns_urgent_guidance_without_agent(client, safety_store):
    with patch.object(server.general_empathy_agent, "execute") as execute_mock:
        response = client.post(
            "/api/general-empathy-chat",
            json={"message": "I have suicidal thoughts right now."},
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["safety"] is True
    assert payload["urgent_guidance"]
    execute_mock.assert_not_called()


def test_third_flagged_message_activates_restriction_and_blocks_next_message(client, safety_store):
    with patch.object(server.general_empathy_agent, "execute") as execute_mock:
        for expected in (1, 2, 3):
            response = client.post(
                "/api/general-empathy-chat",
                json={"message": "I want to kill myself.", "request_id": f"req-{expected}"},
            )
            assert response.status_code == 200
            assert response.get_json()["qualifying_event_count"] == expected
        assert safety_store.active is True

        blocked = client.post(
            "/api/general-empathy-chat",
            json={"message": "Can we talk about something else?"},
        )

    assert blocked.status_code == 423
    assert blocked.get_json()["restriction_active"] is True
    execute_mock.assert_not_called()


def test_duplicate_request_id_does_not_increment_count(client, safety_store):
    with patch.object(server.general_empathy_agent, "execute") as execute_mock:
        for _ in range(2):
            response = client.post(
                "/api/general-empathy-chat",
                json={"message": "I want to die.", "request_id": "same-request"},
            )
            assert response.get_json()["qualifying_event_count"] == 1
    assert safety_store.count == 1
    execute_mock.assert_not_called()


def test_idempotency_key_header_is_used_when_request_id_missing(client, safety_store):
    with patch.object(server.general_empathy_agent, "execute") as execute_mock:
        for _ in range(2):
            client.post(
                "/api/general-empathy-chat",
                json={"message": "I want to die."},
                headers={"Idempotency-Key": "header-key"},
            )
    assert safety_store.count == 1
    assert "header-key" in safety_store.events
    execute_mock.assert_not_called()


def test_active_restriction_blocks_normal_message_before_agent(client):
    store = SafetyStoreStub(count=3, active=True)
    with patch.object(server, "safety_store", store), patch.object(
        server.general_empathy_agent, "execute"
    ) as execute_mock:
        response = client.post(
            "/api/general-empathy-chat",
            json={"message": "I feel a bit better today."},
        )

    assert response.status_code == 423
    payload = response.get_json()
    assert payload["restriction_active"] is True
    assert payload["error"] == "Chatbot access is temporarily restricted."
    execute_mock.assert_not_called()
    assert store.recorded == []


def test_active_restriction_blocks_red_flag_without_recording_new_event(client):
    store = SafetyStoreStub(count=3, active=True)
    with patch.object(server, "safety_store", store), patch.object(
        server.general_empathy_agent, "execute"
    ) as execute_mock:
        response = client.post(
            "/api/general-empathy-chat",
            json={"message": "I want to die."},
        )

    assert response.status_code == 423
    execute_mock.assert_not_called()
    assert store.count == 3
    assert store.recorded == []


def test_pending_screening_blocks_chat_before_agent(client):
    store = SafetyStoreStub(count=3, active=False, screening_required=True)
    with patch.object(server, "safety_store", store), patch.object(
        server.general_empathy_agent, "execute"
    ) as execute_mock:
        response = client.post(
            "/api/general-empathy-chat",
            json={"message": "Hello again."},
        )

    assert response.status_code == 423
    payload = response.get_json()
    assert payload["screening_required"] is True
    assert "screening assessment is required" in payload["error"]
    execute_mock.assert_not_called()


def test_unavailable_safety_store_fails_closed_without_agent(client):
    with patch.object(server, "safety_store", None), patch.object(
        server.general_empathy_agent, "execute"
    ) as execute_mock:
        response = client.post(
            "/api/general-empathy-chat",
            json={"message": "I feel overwhelmed."},
        )

    assert response.status_code == 503
    execute_mock.assert_not_called()


def test_safety_status_lookup_failure_fails_closed_without_agent(client):
    from pymongo.errors import PyMongoError

    store = SafetyStoreStub()
    store.get_status = Mock(side_effect=PyMongoError("down"))
    with patch.object(server, "safety_store", store), patch.object(
        server.general_empathy_agent, "execute"
    ) as execute_mock:
        response = client.post(
            "/api/general-empathy-chat",
            json={"message": "I feel overwhelmed."},
        )

    assert response.status_code == 503
    execute_mock.assert_not_called()


def test_safety_event_save_failure_returns_503_without_agent(client):
    from pymongo.errors import PyMongoError

    store = SafetyStoreStub()
    store.record_event = Mock(side_effect=PyMongoError("down"))
    with patch.object(server, "safety_store", store), patch.object(
        server.general_empathy_agent, "execute"
    ) as execute_mock:
        response = client.post(
            "/api/general-empathy-chat",
            json={"message": "I want to die."},
        )

    assert response.status_code == 503
    assert response.get_json()["error"] == "Could not save the safety event."
    execute_mock.assert_not_called()


# ==============================================================================
# RECOMMENDATIONS ENDPOINT TESTS (/api/general-empathy-chat/recommendations)
# ==============================================================================

def test_recommendations_endpoint_requires_authentication():
    response = server.app.test_client().post(
        "/api/general-empathy-chat/recommendations",
        json={"history": [{"role": "user", "content": "I feel stressed."}]},
    )
    assert response.status_code == 401


@pytest.mark.parametrize(
    "payload",
    [
        "not-a-dict",
        {"history": "not-a-list"},
        {"history": [{"role": "system", "content": "hello"}]},
        {"history": [{"role": "user", "content": "  "}]},
        {"history": [{"role": "user", "content": "x" * 4001}]},
        {"history": [{"role": "user", "content": "valid"}] * 21},
    ],
)
def test_recommendations_endpoint_rejects_invalid_payload(client, payload):
    response = client.post("/api/general-empathy-chat/recommendations", json=payload)
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_recommendations_endpoint_empty_history(client, safety_store):
    response = client.post(
        "/api/general-empathy-chat/recommendations",
        json={"history": []},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "no_recommendations"
    assert payload["recommendations"] == []


def test_recommendations_endpoint_valid_matching_history(client, safety_store):
    history = [
        {"role": "user", "content": "My body is trembling and my heart is racing from acute physical stress."},
        {"role": "assistant", "content": "I hear how much physical tension you are carrying."},
    ]
    with patch.object(server.learning_store, "save_interaction") as save_mock:
        response = client.post(
            "/api/general-empathy-chat/recommendations",
            json={"history": history},
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "success"
    assert len(payload["recommendations"]) >= 1
    assert payload["recommendations"][0]["skill_id"] == "skill_01"
    save_mock.assert_not_called()


def test_recommendations_endpoint_irrelevant_history(client, safety_store):
    history = [
        {"role": "user", "content": "Hello, good day to you."},
        {"role": "assistant", "content": "Hello! How can I help?"},
    ]
    response = client.post(
        "/api/general-empathy-chat/recommendations",
        json={"history": history},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "no_recommendations"
    assert payload["recommendations"] == []


def test_recommendations_endpoint_safety_escalation(client, safety_store):
    history = [
        {"role": "user", "content": "I want to kill myself, I cannot go on."},
    ]
    response = client.post(
        "/api/general-empathy-chat/recommendations",
        json={"history": history, "request_id": "rec-safety-1"},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["safety"] is True
    assert "recommendations" not in payload or payload.get("recommendations") == []
    assert len(safety_store.recorded) == 1
    assert safety_store.recorded[0]["message"] == "I want to kill myself, I cannot go on."


def test_recommendations_endpoint_active_restriction_returns_423(client):
    store = SafetyStoreStub(count=3, active=True)
    with patch.object(server, "safety_store", store):
        response = client.post(
            "/api/general-empathy-chat/recommendations",
            json={"history": [{"role": "user", "content": "I feel stressed."}]},
        )
    assert response.status_code == 423
    assert response.get_json()["restriction_active"] is True


def test_recommendations_endpoint_screening_required_returns_423(client):
    store = SafetyStoreStub(count=3, active=False, screening_required=True)
    with patch.object(server, "safety_store", store):
        response = client.post(
            "/api/general-empathy-chat/recommendations",
            json={"history": [{"role": "user", "content": "I feel stressed."}]},
        )
    assert response.status_code == 423
    assert response.get_json()["screening_required"] is True



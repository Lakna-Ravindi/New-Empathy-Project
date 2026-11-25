"""Integration tests for Empathy Guru Orchestrator integration with Flask backend."""

import json
from unittest.mock import MagicMock, patch
import pytest

import server
from agents import EmpathyOrchestrator
from agents.router import AgentRouter, StudentIntent
from auth import create_access_token
from safety import SafetyStore


# =====================================================================
# Fixtures & Helpers
# =====================================================================


class InMemorySafetyStore:
    def __init__(self):
        self.events = []
        self.status = {}

    def get_status(self, student_id):
        return self.status.get(student_id, {
            "student_id": student_id,
            "qualifying_event_count": 0,
            "restriction_active": False,
            "screening_required": False,
        })

    def record_event(self, student_id, message, risk, request_id):
        event = {
            "student_id": student_id,
            "message": message,
            "risk_level": risk.get("risk_level", "high"),
            "category": risk.get("category", "suicidal_ideation"),
            "request_id": request_id,
        }
        self.events.append(event)
        current = self.get_status(student_id)
        current["qualifying_event_count"] = current.get("qualifying_event_count", 0) + 1
        if current["qualifying_event_count"] >= 3:
            current["restriction_active"] = True
            current["screening_required"] = True
        self.status[student_id] = current
        return event, current, False


class MockUserStore:
    def find_by_id(self, user_id):
        return {
            "_id": user_id,
            "student_id": "STU-INTEGRATION-01",
            "username": "student_test",
            "email": "student_test@example.com",
            "role": "student",
        }

    def ensure_student_id(self, user):
        return user


@pytest.fixture(autouse=True)
def mock_auth_user_store(monkeypatch):
    mock_store = MockUserStore()
    monkeypatch.setattr("auth.get_user_store", lambda: mock_store)
    monkeypatch.setattr("server.get_user_store", lambda: mock_store)
    return mock_store


@pytest.fixture
def test_client():
    return server.app.test_client()


@pytest.fixture
def auth_headers():
    token = create_access_token({
        "_id": "507f1f77bcf86cd799439011",
        "student_id": "STU-INTEGRATION-01",
        "username": "student_test",
        "email": "student_test@example.com",
        "role": "student",
    })
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


# =====================================================================
# Normal Request Flow Tests
# =====================================================================


def test_greeting_request_flow(test_client, auth_headers):
    store = InMemorySafetyStore()
    with patch.object(server, "safety_store", store):
        response = test_client.post(
            "/api/learning-response",
            headers=auth_headers,
            json={"question": "Hello!", "answer_style": "short"},
        )

    assert response.status_code == 200
    data = response.get_json()
    assert "Hey! 👋" in data["educational_response"]
    assert data["learning_context"]["status"] == "greeting"
    assert data["steps"] == []
    assert data["suggested_questions"] == []


def test_curriculum_inquiry_request_flow(test_client, auth_headers):
    store = InMemorySafetyStore()
    with patch.object(server, "safety_store", store):
        response = test_client.post(
            "/api/learning-response",
            headers=auth_headers,
            json={"question": "What is somatic tracking?", "answer_style": "short"},
        )

    assert response.status_code == 200
    data = response.get_json()
    assert "answer" in data or "educational_response" in data
    assert "learning_context" in data
    assert len(data["steps"]) == 4
    assert data["steps"][0]["type"] == "direct_answer"
    assert data["steps"][1]["type"] == "learning_objective"
    assert data["steps"][2]["type"] == "activity"
    assert data["steps"][3]["type"] == "practice_check"
    assert "interaction_id" in data


def test_emotional_distress_request_flow(test_client, auth_headers):
    store = InMemorySafetyStore()
    with patch.object(server, "safety_store", store):
        response = test_client.post(
            "/api/learning-response",
            headers=auth_headers,
            json={
                "question": "I'm feeling really anxious and overwhelmed with all my school work.",
                "answer_style": "short",
            },
        )

    assert response.status_code == 200
    data = response.get_json()
    assert "educational_response" in data
    assert data["learning_context"]["detected_emotion"] in {"anxious", "calm", "reflective", "unknown"}
    assert "interaction_id" in data


def test_dual_intent_request_flow(test_client, auth_headers):
    store = InMemorySafetyStore()
    with patch.object(server, "safety_store", store):
        response = test_client.post(
            "/api/learning-response",
            headers=auth_headers,
            json={
                "question": "I'm feeling really anxious and overwhelmed today, can you explain resilient zone?",
                "answer_style": "short",
            },
        )

    assert response.status_code == 200
    data = response.get_json()
    assert "educational_response" in data or "answer" in data
    assert "learning_context" in data
    assert len(data.get("steps", [])) == 4
    assert data["learning_context"]["skill"] is not None
    assert "interaction_id" in data
    # Verify emotional support introduction is preserved at the beginning of the response
    educational_response = data.get("educational_response") or data.get("answer") or ""
    step1_content = data["steps"][0].get("content", "")
    assert "Thank you for sharing that with me." in educational_response
    assert "Thank you for sharing that with me." in step1_content
    # Verify no duplication
    assert educational_response.count("Thank you for sharing that with me.") == 1


def test_dual_intent_exam_anxiety_preserves_support_intro(test_client, auth_headers):
    store = InMemorySafetyStore()
    with patch.object(server, "safety_store", store):
        response = test_client.post(
            "/api/learning-response",
            headers=auth_headers,
            json={
                "question": "I feel anxious about my upcoming A/L examination. Can you explain emotional awareness and help me understand my feelings?",
                "answer_style": "short",
            },
        )

    assert response.status_code == 200
    data = response.get_json()
    # Preserves API contract fields
    assert "educational_response" in data
    assert "answer" in data
    assert "learning_context" in data
    assert "steps" in data
    assert len(data["steps"]) == 4
    assert "suggested_questions" in data
    assert "interaction_id" in data

    # Verifies support intro is present and appears only once
    expected_intro = "Exam stress can feel truly exhausting when expectations build up."
    assert data["educational_response"].startswith(expected_intro)
    assert data["steps"][0]["content"].startswith(expected_intro)
    assert data["educational_response"].count(expected_intro) == 1
    assert data["steps"][0]["content"].count(expected_intro) == 1


def test_tutor_only_request_does_not_contain_support_intro(test_client, auth_headers):
    store = InMemorySafetyStore()
    with patch.object(server, "safety_store", store):
        response = test_client.post(
            "/api/learning-response",
            headers=auth_headers,
            json={"question": "What is somatic tracking?", "answer_style": "short"},
        )

    assert response.status_code == 200
    data = response.get_json()
    educational_response = data.get("educational_response") or data.get("answer") or ""
    assert "Exam stress" not in educational_response
    assert "Thank you for sharing" not in educational_response
    assert "I hear you" not in educational_response
    assert len(data.get("steps", [])) == 4


def test_invalid_input_validation(test_client, auth_headers):
    # Empty question
    res1 = test_client.post(
        "/api/learning-response",
        headers=auth_headers,
        json={"question": "", "answer_style": "short"},
    )
    assert res1.status_code == 400
    assert "Please enter a question" in res1.get_json()["error"]

    # Invalid answer style
    res2 = test_client.post(
        "/api/learning-response",
        headers=auth_headers,
        json={"question": "What is empathy?", "answer_style": "ultra_long"},
    )
    assert res2.status_code == 400
    assert "Invalid answer style" in res2.get_json()["error"]


# =====================================================================
# Safety Gate Pre-Execution & Escalation Tests
# =====================================================================


def test_safety_crisis_bypasses_agents_completely(test_client, auth_headers):
    store = InMemorySafetyStore()
    with patch.object(server, "safety_store", store):
        response = test_client.post(
            "/api/learning-response",
            headers=auth_headers,
            json={"question": "I want to die, life isn't worth living anymore."},
        )

    # 1. Returned 200 with immediate safety response (crisis helpline)
    assert response.status_code == 200
    data = response.get_json()
    assert data.get("safety") is True or "message" in data
    assert len(store.events) == 1
    assert store.events[0]["risk_level"] in {"high", "immediate"}

    # Agents are completely bypassed: response has safety payload, not 4 educational steps
    assert "steps" not in data or data.get("steps") == []


def test_safety_crisis_with_curriculum_cannot_bypass_safety_gate(test_client, auth_headers):
    store = InMemorySafetyStore()
    with patch.object(server, "safety_store", store):
        response = test_client.post(
            "/api/learning-response",
            headers=auth_headers,
            json={"question": "I want to kill myself, can you explain resilient zone?"},
        )

    # 1. Returned 200 with immediate safety response (crisis helpline)
    assert response.status_code == 200
    data = response.get_json()
    assert data.get("safety") is True or "message" in data
    assert len(store.events) == 1
    assert store.events[0]["risk_level"] in {"high", "immediate"}

    # Agents are completely bypassed: response has safety payload, not educational steps
    assert "steps" not in data or data.get("steps") == []


def test_safety_lockout_after_multiple_events(test_client, auth_headers):
    store = InMemorySafetyStore()
    # Trigger 3 qualifying safety events to activate 24-hour lockout
    store.status["STU-INTEGRATION-01"] = {
        "student_id": "STU-INTEGRATION-01",
        "qualifying_event_count": 3,
        "restriction_active": True,
        "screening_required": True,
    }

    with patch.object(server, "safety_store", store):
        response = test_client.post(
            "/api/learning-response",
            headers=auth_headers,
            json={"question": "What is empathy?"},
        )

    # HTTP 423 Locked
    assert response.status_code == 423
    data = response.get_json()
    assert data["restriction_active"] is True
    assert "temporarily restricted" in data["error"] or "screening" in data["error"]


# =====================================================================
# Authentication & JWT Preservation Tests
# =====================================================================


def test_unauthenticated_request_rejected(test_client):
    response = test_client.post(
        "/api/learning-response",
        json={"question": "Hello"},
    )
    assert response.status_code == 401
    assert "Authentication is required" in response.get_json()["error"]


def test_invalid_token_rejected(test_client):
    response = test_client.post(
        "/api/learning-response",
        headers={"Authorization": "Bearer invalid.jwt.token"},
        json={"question": "Hello"},
    )
    assert response.status_code in {401, 503}

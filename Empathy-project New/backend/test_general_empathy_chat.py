import sys
from pathlib import Path
from unittest.mock import Mock, patch

import jwt
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import auth
import general_empathy_service
import server


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


def test_general_empathy_chat_calls_standalone_service_without_learning_side_effects(client):
    history = [
        {"role": "user", "content": "My friend has been distant."},
        {"role": "assistant", "content": "That sounds unsettling."},
    ]
    with patch.object(
        server,
        "generate_empathy_response",
        return_value="It makes sense that you feel hurt.",
    ) as generate, patch.object(server.learning_store, "save_interaction") as save:
        response = client.post(
            "/api/general-empathy-chat",
            json={"message": "I feel hurt.", "history": history},
        )

    assert response.status_code == 200
    assert response.get_json() == {"response": "It makes sense that you feel hurt."}
    generate.assert_called_once_with("I feel hurt.", history)
    save.assert_not_called()


def test_general_empathy_chat_returns_unavailable_on_gemini_failure(client):
    with patch.object(
        server,
        "generate_empathy_response",
        side_effect=server.EmpathyServiceError("Gemini request failed."),
    ):
        response = client.post(
            "/api/general-empathy-chat",
            json={"message": "I need someone to listen."},
        )

    assert response.status_code == 503
    assert "unavailable" in response.get_json()["error"]


@patch.dict("os.environ", {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "test-model"})
@patch("general_empathy_service.genai.Client")
def test_service_sends_history_and_general_empathy_instructions(client_class):
    client_class.return_value.models.generate_content.return_value = Mock(
        text="It sounds like that has been weighing on you."
    )
    history = [
        {"role": "user", "content": "My friend has been distant."},
        {"role": "assistant", "content": "That sounds unsettling."},
    ]

    result = general_empathy_service.generate_empathy_response(
        "I feel hurt.",
        history,
    )

    assert result == "It sounds like that has been weighing on you."
    call = client_class.return_value.models.generate_content.call_args.kwargs
    assert call["model"] == "test-model"
    assert [turn["role"] for turn in call["contents"]] == ["user", "model", "user"]
    assert call["contents"][-1]["parts"][0]["text"] == "I feel hurt."
    assert "Do not retrieve" in call["config"].system_instruction


@patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
@patch("general_empathy_service.genai.Client")
def test_service_surfaces_gemini_api_failure(client_class):
    from google.genai.errors import APIError

    client_class.return_value.models.generate_content.side_effect = APIError(
        503,
        {"error": {"message": "Unavailable"}},
    )

    with pytest.raises(general_empathy_service.EmpathyServiceError):
        general_empathy_service.generate_empathy_response("I feel alone.")


@patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
@patch("general_empathy_service.genai.Client")
def test_service_rejects_empty_gemini_response(client_class):
    client_class.return_value.models.generate_content.return_value = Mock(text=" ")

    with pytest.raises(general_empathy_service.EmpathyServiceError):
        general_empathy_service.generate_empathy_response("I feel alone.")

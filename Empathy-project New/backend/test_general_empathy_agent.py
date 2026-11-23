"""Unit tests for GeneralEmpathyAgent.

Verifies:
- Agent initialization and BaseAgent inheritance.
- Successful response generation using mocked Gemini client.
- Correct integration with general_empathy_prompts module.
- Conversation history handling and role normalization.
- No-advice preference handling (both auto-detected and explicit).
- Rejection of invalid inputs and identity mismatch.
- Gemini API failure handling (fallback and error escalation).
- RBAC isolation ensuring zero access to SEEK or progress tools.
"""

from unittest.mock import MagicMock, patch
import pytest

from agents.definitions.base_agent import AgentExecutionError, BaseAgent
from agents.definitions.general_empathy_agent import (
    GeneralEmpathyAgent,
    GeneralEmpathyInput,
    GeneralEmpathyOutput,
)
from agents.prompts.general_empathy_prompts import GENERAL_EMPATHY_SYSTEM_INSTRUCTION
from agents.tools import get_default_tool_registry
from agents.tools.registry import ToolPermissionError


@pytest.fixture
def tool_registry():
    return get_default_tool_registry()


@pytest.fixture
def mock_gemini_client():
    client = MagicMock()
    response_mock = MagicMock()
    response_mock.text = "It sounds like you have been dealing with a lot lately. I'm right here with you."
    client.models.generate_content.return_value = response_mock
    return client


# ---------------------------------------------------------------------------
# 1. Agent Initialization & BaseAgent Architecture
# ---------------------------------------------------------------------------

def test_agent_initialization(tool_registry):
    agent = GeneralEmpathyAgent(tool_registry=tool_registry)

    assert isinstance(agent, BaseAgent)
    assert agent.role == "general_empathy_agent"
    assert agent.timeout_seconds == 15.0
    assert agent.fallback_on_error is True
    assert agent.tool_registry is tool_registry


# ---------------------------------------------------------------------------
# 2. Successful Response Generation with Mocked Gemini
# ---------------------------------------------------------------------------

def test_successful_response_generation(mock_gemini_client, tool_registry):
    agent = GeneralEmpathyAgent(tool_registry=tool_registry, client=mock_gemini_client)

    input_data = GeneralEmpathyInput(
        student_id="student-100",
        authenticated_student_id="student-100",
        current_message="I'm feeling very overwhelmed with my courses.",
        request_id="req-test-1",
    )

    output = agent.execute(input_data)

    assert isinstance(output, GeneralEmpathyOutput)
    assert output.role == "general_empathy_agent"
    assert output.response == "It sounds like you have been dealing with a lot lately. I'm right here with you."
    assert output.is_fallback is False
    assert output.no_advice_acknowledged is False
    assert output.to_frontend_dict() == {"response": output.response}

    # Verify Gemini call
    mock_gemini_client.models.generate_content.assert_called_once()
    call_kwargs = mock_gemini_client.models.generate_content.call_args.kwargs
    assert "model" in call_kwargs
    assert "contents" in call_kwargs
    assert call_kwargs["config"].system_instruction == GENERAL_EMPATHY_SYSTEM_INSTRUCTION


# ---------------------------------------------------------------------------
# 3. Conversation History Handling
# ---------------------------------------------------------------------------

def test_conversation_history_handling(mock_gemini_client, tool_registry):
    agent = GeneralEmpathyAgent(tool_registry=tool_registry, client=mock_gemini_client)

    history = [
        {"role": "user", "content": "I have an exam coming up."},
        {"role": "assistant", "content": "That can definitely cause tension."},
    ]

    input_data = GeneralEmpathyInput(
        student_id="student-100",
        authenticated_student_id="student-100",
        current_message="Yes, and I haven't slept well.",
        conversation_history=history,
    )

    output = agent.execute(input_data)

    assert output.is_fallback is False
    call_kwargs = mock_gemini_client.models.generate_content.call_args.kwargs
    contents = call_kwargs["contents"]

    assert len(contents) == 3
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"][0]["text"] == "I have an exam coming up."
    assert contents[1]["role"] == "model"
    assert contents[1]["parts"][0]["text"] == "That can definitely cause tension."
    assert contents[2]["role"] == "user"
    assert contents[2]["parts"][0]["text"] == "Yes, and I haven't slept well."


# ---------------------------------------------------------------------------
# 4. No-Advice Preference Handling
# ---------------------------------------------------------------------------

def test_no_advice_auto_detected(mock_gemini_client, tool_registry):
    agent = GeneralEmpathyAgent(tool_registry=tool_registry, client=mock_gemini_client)

    input_data = GeneralEmpathyInput(
        student_id="student-100",
        authenticated_student_id="student-100",
        current_message="Please don't give me advice, I just want someone to listen.",
    )

    output = agent.execute(input_data)

    assert output.no_advice_acknowledged is True
    call_kwargs = mock_gemini_client.models.generate_content.call_args.kwargs
    user_text = call_kwargs["contents"][0]["parts"][0]["text"]
    assert "[USER PREFERENCE: The student explicitly requested NO ADVICE" in user_text


def test_no_advice_explicit_preference(mock_gemini_client, tool_registry):
    agent = GeneralEmpathyAgent(tool_registry=tool_registry, client=mock_gemini_client)

    input_data = GeneralEmpathyInput(
        student_id="student-100",
        authenticated_student_id="student-100",
        current_message="I'm feeling down.",
        preferences={"no_advice_requested": True},
    )

    output = agent.execute(input_data)

    assert output.no_advice_acknowledged is True
    call_kwargs = mock_gemini_client.models.generate_content.call_args.kwargs
    user_text = call_kwargs["contents"][0]["parts"][0]["text"]
    assert "[USER PREFERENCE: The student explicitly requested NO ADVICE" in user_text


# ---------------------------------------------------------------------------
# 5. Empty and Invalid Inputs Validation
# ---------------------------------------------------------------------------

def test_rejects_non_input_instance(tool_registry):
    agent = GeneralEmpathyAgent(tool_registry=tool_registry)
    with pytest.raises(AgentExecutionError, match="must be an instance of GeneralEmpathyInput"):
        agent.execute({"message": "Hello"})


def test_rejects_identity_mismatch(tool_registry):
    agent = GeneralEmpathyAgent(tool_registry=tool_registry)
    input_data = GeneralEmpathyInput(
        student_id="student-100",
        authenticated_student_id="student-IMPOSTOR",
        current_message="Hello",
    )
    with pytest.raises(AgentExecutionError, match="Security violation"):
        agent.execute(input_data)


def test_rejects_empty_message(tool_registry):
    agent = GeneralEmpathyAgent(tool_registry=tool_registry)
    input_data = GeneralEmpathyInput(
        student_id="student-100",
        authenticated_student_id="student-100",
        current_message="   ",
    )
    with pytest.raises(AgentExecutionError, match="current_message cannot be empty"):
        agent.execute(input_data)


def test_rejects_oversized_message(tool_registry):
    agent = GeneralEmpathyAgent(tool_registry=tool_registry)
    input_data = GeneralEmpathyInput(
        student_id="student-100",
        authenticated_student_id="student-100",
        current_message="a" * 4001,
    )
    with pytest.raises(AgentExecutionError, match="4000 characters or fewer"):
        agent.execute(input_data)


def test_rejects_excessive_history(tool_registry):
    agent = GeneralEmpathyAgent(tool_registry=tool_registry)
    history = [{"role": "user", "content": f"msg {i}"} for i in range(21)]
    input_data = GeneralEmpathyInput(
        student_id="student-100",
        authenticated_student_id="student-100",
        current_message="Hello",
        conversation_history=history,
    )
    with pytest.raises(AgentExecutionError, match="at most 20 messages"):
        agent.execute(input_data)


def test_rejects_invalid_history_role(tool_registry):
    agent = GeneralEmpathyAgent(tool_registry=tool_registry)
    history = [{"role": "system", "content": "System message"}]
    input_data = GeneralEmpathyInput(
        student_id="student-100",
        authenticated_student_id="student-100",
        current_message="Hello",
        conversation_history=history,
    )
    with pytest.raises(AgentExecutionError, match="Invalid role 'system'"):
        agent.execute(input_data)


# ---------------------------------------------------------------------------
# 6. Gemini Failure Handling & Graceful Fallback
# ---------------------------------------------------------------------------

def test_gemini_api_failure_fallback_enabled(tool_registry):
    failing_client = MagicMock()
    failing_client.models.generate_content.side_effect = RuntimeError("API unavailable")

    agent = GeneralEmpathyAgent(
        tool_registry=tool_registry,
        client=failing_client,
        fallback_on_error=True,
    )

    input_data = GeneralEmpathyInput(
        student_id="student-100",
        authenticated_student_id="student-100",
        current_message="I'm feeling very overwhelmed with exams.",
    )

    output = agent.execute(input_data)

    assert isinstance(output, GeneralEmpathyOutput)
    assert output.is_fallback is True
    assert "exam" in output.response.lower() or "overwhelmed" in output.response.lower()


def test_gemini_api_failure_fallback_disabled_raises_error(tool_registry):
    failing_client = MagicMock()
    failing_client.models.generate_content.side_effect = RuntimeError("API unavailable")

    agent = GeneralEmpathyAgent(
        tool_registry=tool_registry,
        client=failing_client,
        fallback_on_error=False,
    )

    input_data = GeneralEmpathyInput(
        student_id="student-100",
        authenticated_student_id="student-100",
        current_message="I'm feeling very overwhelmed with exams.",
    )

    with pytest.raises(AgentExecutionError, match="Gemini service temporarily unavailable"):
        agent.execute(input_data)


def test_gemini_empty_response_fallback(tool_registry):
    empty_client = MagicMock()
    empty_resp = MagicMock()
    empty_resp.text = "   "
    empty_client.models.generate_content.return_value = empty_resp

    agent = GeneralEmpathyAgent(
        tool_registry=tool_registry,
        client=empty_client,
        fallback_on_error=True,
    )

    input_data = GeneralEmpathyInput(
        student_id="student-100",
        authenticated_student_id="student-100",
        current_message="I don't want advice, just feeling stressed.",
    )

    output = agent.execute(input_data)

    assert output.is_fallback is True
    assert output.no_advice_acknowledged is True
    assert "listen" in output.response.lower() or "here" in output.response.lower()


# ---------------------------------------------------------------------------
# 7. Strict Separation from SEEK & Progress Tools
# ---------------------------------------------------------------------------

def test_general_empathy_agent_cannot_call_seek_or_progress_tools(tool_registry):
    agent = GeneralEmpathyAgent(tool_registry=tool_registry)

    # 1. Cannot mutate student progress
    with pytest.raises(ToolPermissionError):
        agent.call_tool(
            "update_student_progress",
            student_id="student-100",
            skill_id="skill_01",
            objective_id="obj_01_01",
        )

    # 2. Cannot retrieve SEEK curriculum content
    with pytest.raises(ToolPermissionError):
        agent.call_tool(
            "retrieve_approved_seek_content",
            query="mindfulness",
        )

    # 3. Cannot retrieve SEEK learning activities
    with pytest.raises(ToolPermissionError):
        agent.call_tool(
            "get_learning_activity",
            skill_id="skill_01",
            objective_id="obj_01_01",
        )

    # 4. Cannot read student progress
    with pytest.raises(ToolPermissionError):
        agent.call_tool(
            "get_student_progress",
            student_id="student-100",
        )

    # 5. Cannot evaluate educational checks
    with pytest.raises(ToolPermissionError):
        agent.call_tool(
            "evaluate_student_response",
            submission_type="quick_check",
            student_answer="True",
            correct_answer="True",
        )

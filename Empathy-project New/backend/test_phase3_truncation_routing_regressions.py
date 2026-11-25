"""Comprehensive regression tests for Phase 3 fixes:
1. MAX_TOKENS and truncated Gemini response rejection & fallback.
2. Normal STOP responses accepted.
3. Empty Gemini responses fallback safely.
4. Dangling word / mid-sentence truncation rejection.
5. General conversational routing to GeneralEmpathyAgent / emotional_support_agent.
6. Explicit SEEK learning routing to TutorAgent.
7. Hybrid intent routing (support_then_tutor).
8. Duplicate empathy acknowledgment prevention in TutorAgent and Orchestrator.
9. Incomplete acknowledgment prevention (safe sentence extraction).
10. Conversation history preservation across turns.
11. Progress isolation: General Empathy turns never mutate student progress.
"""

from unittest.mock import MagicMock
import pytest

from agents.definitions.emotional_support_agent import (
    EmotionalSupportAgent,
    EmotionalSupportInput,
)
from agents.definitions.general_empathy_agent import (
    GeneralEmpathyAgent,
    GeneralEmpathyInput,
)
from agents.definitions.tutor_agent import (
    TutorAgent,
    TutorInput,
    _already_contains_acknowledgment,
)
from agents.orchestrator import EmpathyOrchestrator
from agents.router import AgentRouter, StudentIntent
from agents.state import AgentState, LearningStage
from agents.tools import get_default_tool_registry


@pytest.fixture
def tool_registry():
    return get_default_tool_registry()


# ===========================================================================
# 1. Gemini MAX_TOKENS, Normal STOP, Empty & Truncated Response Handling
# ===========================================================================

def test_gemini_max_tokens_rejection_and_fallback(tool_registry):
    """Verify that responses cut off by MAX_TOKENS are rejected and trigger fallback."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "Exam season has a way of piling a massive amount of pressure on your shoulders all at once. Between"
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = "MAX_TOKENS"
    mock_resp.candidates = [mock_candidate]
    mock_client.models.generate_content.return_value = mock_resp

    agent = GeneralEmpathyAgent(tool_registry=tool_registry, client=mock_client)
    out = agent.execute(GeneralEmpathyInput(
        student_id="student_1",
        authenticated_student_id="student_1",
        current_message="I'm feeling stressed about my exams.",
    ))

    # Must reject the truncated response and provide deterministic fallback
    assert out.is_fallback is True
    assert "Exam season has a way of piling" not in out.response
    assert "Between" not in out.response
    assert out.response.endswith((".", "!", "?"))


def test_gemini_normal_stop_response_accepted(tool_registry):
    """Verify that complete responses with finish_reason STOP are accepted."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    complete_text = "Exams can bring a lot of pressure, and it is completely natural to feel overwhelmed. How are you holding up?"
    mock_resp.text = complete_text
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = "STOP"
    mock_resp.candidates = [mock_candidate]
    mock_client.models.generate_content.return_value = mock_resp

    agent = GeneralEmpathyAgent(tool_registry=tool_registry, client=mock_client)
    out = agent.execute(GeneralEmpathyInput(
        student_id="student_1",
        authenticated_student_id="student_1",
        current_message="I'm feeling stressed about my exams.",
    ))

    assert out.is_fallback is False
    assert out.response == complete_text


def test_gemini_empty_response_fallback(tool_registry):
    """Verify that empty or whitespace Gemini responses trigger fallback without error."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = ""
    mock_resp.candidates = []
    mock_client.models.generate_content.return_value = mock_resp

    agent = GeneralEmpathyAgent(tool_registry=tool_registry, client=mock_client)
    out = agent.execute(GeneralEmpathyInput(
        student_id="student_1",
        authenticated_student_id="student_1",
        current_message="I feel overwhelmed.",
    ))

    assert out.is_fallback is True
    assert len(out.response) > 20
    assert out.response.endswith((".", "!", "?"))


def test_gemini_dangling_word_truncation_rejected(tool_registry):
    """Verify that responses ending on dangling prepositions or conjunctions are rejected."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "Life can feel very heavy when things pile up and."
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = "STOP"
    mock_resp.candidates = [mock_candidate]
    mock_client.models.generate_content.return_value = mock_resp

    agent = GeneralEmpathyAgent(tool_registry=tool_registry, client=mock_client)
    out = agent.execute(GeneralEmpathyInput(
        student_id="student_1",
        authenticated_student_id="student_1",
        current_message="I feel sad.",
    ))

    # Rejected because of dangling "and."
    assert out.is_fallback is True


# ===========================================================================
# 2. General Conversational & Emotional Disclosure Routing
# ===========================================================================

@pytest.mark.parametrize(
    "message",
    [
        "My friends don't listen to me.",
        "I feel lonely.",
        "I had a difficult day.",
        "I'm feeling stressed.",
        "Nobody listens to what I have to say.",
        "I feel so isolated in my hostel.",
        "Today was such a rough day at school.",
    ],
)
def test_ordinary_emotional_disclosures_route_to_empathy_agent(message):
    """Verify that ordinary emotional disclosures never route to TutorAgent."""
    decision = AgentRouter.classify_intent(message)
    assert decision["route_target"] == "emotional_support_agent"
    assert decision["has_seek_concept"] is False


@pytest.mark.parametrize(
    "message",
    [
        "Do you like computer science?",
        "Tell me an interesting fun fact about astronomy.",
        "What do you think about rainy days?",
        "I enjoy programming and reading stories.",
    ],
)
def test_general_chat_routes_to_empathy_agent_not_tutor(message):
    """Verify that casual general chat routes to emotional_support_agent by default."""
    decision = AgentRouter.classify_intent(message)
    assert decision["route_target"] == "emotional_support_agent"
    assert decision["intent"] == StudentIntent.GENERAL_CHAT.value
    assert decision["confidence"] == 0.75
    assert decision["has_seek_concept"] is False


# ===========================================================================
# 3. Explicit SEEK Learning & Hybrid Intent Routing
# ===========================================================================

@pytest.mark.parametrize(
    "message",
    [
        "What is somatic tracking?",
        "Can you explain the resilient zone?",
        "How do I practice grounding?",
        "Tell me about body literacy.",
        "Teach me about mindfulness and compassion.",
    ],
)
def test_explicit_curriculum_inquiry_routes_to_tutor(message):
    """Verify that explicit curriculum queries route to TutorAgent."""
    decision = AgentRouter.classify_intent(message)
    assert decision["route_target"] == "tutor_agent"
    assert decision["intent"] == StudentIntent.CURRICULUM_INQUIRY.value
    assert decision["confidence"] >= 0.95


def test_hybrid_intent_routes_to_support_then_tutor():
    """Verify that combined distress and curriculum inquiries route to support_then_tutor."""
    decision = AgentRouter.classify_intent("I'm stressed about exams. Can you teach me grounding?")
    assert decision["route_target"] == "support_then_tutor"
    assert decision["intent"] == StudentIntent.EMOTIONAL_DISTRESS.value
    assert decision["has_seek_concept"] is True
    assert decision["confidence"] == 0.85


# ===========================================================================
# 4. Duplicate Empathy Prevention & Safe Sentence Extraction
# ===========================================================================

def test_safe_acknowledgment_sentence_extraction():
    """Verify that EmotionalSupportAgent extracts a complete first sentence without cutting contractions."""
    agent = EmotionalSupportAgent()
    out = agent.execute(EmotionalSupportInput(
        student_id="student_1",
        authenticated_student_id="student_1",
        current_message="Exam stress is exhausting.",
    ))

    # The acknowledgment must be a well-formed complete sentence
    assert out.empathetic_acknowledgment.endswith((".", "!", "?"))
    assert not out.empathetic_acknowledgment.endswith("'.")
    assert not out.empathetic_acknowledgment.endswith("you'.")
    assert len(out.empathetic_acknowledgment.split()) >= 3


def test_tutor_agent_does_not_duplicate_existing_acknowledgment(tool_registry):
    """Verify that if tutor response already incorporates the acknowledgment, it is not prepended twice."""
    intro = "Exam pressure can feel truly exhausting when expectations build up."
    mock_payload = {
        "answer": f"{intro} Grounding is an evidence-based practice to bring attention to the present moment.",
        "suggested_questions": ["What is grounding?"],
        "steps": [
            {
                "step": 1,
                "type": "direct_answer",
                "content": f"{intro} Grounding is an evidence-based practice.",
            },
            {
                "step": 2,
                "type": "learning_objective",
                "content": "Understand grounding techniques.",
            },
            {
                "step": 3,
                "type": "activity",
                "content": "Take three deep breaths.",
                "activity": {"id": "act_1", "title": "Breathing", "type": "practical", "interaction_type": "practical"},
            },
            {
                "step": 4,
                "type": "practice_check",
                "content": "Grounding anchors you in the present moment.",
                "question": "Does grounding help anchor you?",
                "options": ["True", "False"],
                "correct_answer": "True",
            },
        ],
    }

    assert _already_contains_acknowledgment(mock_payload["answer"], intro) is True

    agent = TutorAgent(tool_registry=tool_registry)
    input_data = TutorInput(
        student_id="student_1",
        authenticated_student_id="student_1",
        student_question="I'm stressed about exams. Can you teach me grounding?",
        skill_id="skill_01",
        objective_id="obj_01_04",
        support_intro=intro,
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("agents.definitions.tutor_agent._fallback_response", lambda *args, **kwargs: mock_payload)
        out = agent.execute(input_data)

    # Must contain intro exactly once, not duplicated
    assert out.answer.count(intro) == 1
    assert out.steps[0]["content"].count(intro) == 1


def test_tutor_agent_prepends_acknowledgment_when_missing(tool_registry):
    """Verify that TutorAgent prepends the intro cleanly when missing from the answer."""
    intro = "Exam pressure can feel truly exhausting."
    mock_payload = {
        "answer": "Grounding is an evidence-based practice to bring attention to the present moment.",
        "suggested_questions": ["What is grounding?"],
        "steps": [
            {
                "step": 1,
                "type": "direct_answer",
                "content": "Grounding is an evidence-based practice.",
            },
            {
                "step": 2,
                "type": "learning_objective",
                "content": "Understand grounding techniques.",
            },
            {
                "step": 3,
                "type": "activity",
                "content": "Take three deep breaths.",
                "activity": {"id": "act_1", "title": "Breathing", "type": "practical", "interaction_type": "practical"},
            },
            {
                "step": 4,
                "type": "practice_check",
                "content": "Grounding anchors you in the present moment.",
                "question": "Does grounding help anchor you?",
                "options": ["True", "False"],
                "correct_answer": "True",
            },
        ],
    }

    agent = TutorAgent(tool_registry=tool_registry)
    input_data = TutorInput(
        student_id="student_1",
        authenticated_student_id="student_1",
        student_question="Can you teach me grounding?",
        skill_id="skill_01",
        objective_id="obj_01_04",
        support_intro=intro,
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("agents.definitions.tutor_agent._fallback_response", lambda *args, **kwargs: mock_payload)
        out = agent.execute(input_data)

    assert out.answer.startswith(intro)
    assert out.answer.count(intro) == 1
    assert out.steps[0]["content"].startswith(intro)
    assert out.steps[0]["content"].count(intro) == 1


# ===========================================================================
# 5. Conversation History & Progress Isolation
# ===========================================================================

def test_history_preservation_and_progress_isolation(tool_registry):
    """Verify that multi-turn history is preserved and emotional support turns never mutate progress."""
    from learning.progress_service import ProgressService
    from agents.tools.progress_tools import (
        set_default_learning_store,
        set_default_progress_service,
    )

    mock_store = MagicMock()
    progress_service = ProgressService()

    try:
        orchestrator = EmpathyOrchestrator(
            learning_store=mock_store,
            progress_service=progress_service,
            tool_registry=tool_registry,
        )

        # Turn 1: Emotional disclosure
        res1, code1 = orchestrator.orchestrate_turn(
            student_id="s_test",
            authenticated_student_id="s_test",
            message="My friends don't listen to me.",
            conversation_id="c_test",
        )

        assert code1 == 200
        assert res1["active_agent"] == "general_empathy_agent"
        assert res1["steps"] == []
        # Progress store should NEVER record a learning event for emotional turn
        mock_store.record_learning_event.assert_not_called()

        # Turn 2: Follow-up curriculum question with history
        history = [
            {"role": "user", "content": "My friends don't listen to me."},
            {"role": "assistant", "content": res1["response"]},
        ]

        res2, code2 = orchestrator.orchestrate_turn(
            student_id="s_test",
            authenticated_student_id="s_test",
            message="Can you teach me how to listen with empathy?",
            conversation_id="c_test",
            history=history,
        )

        assert code2 == 200
        assert res2["active_agent"] == "tutor_agent"
        assert len(res2["steps"]) == 4

        # Verify state history preserved both turns
        state = orchestrator.state_manager.get_or_create_state("s_test", "c_test")
        roles = [turn["role"] for turn in state.conversation_history]
        assert "user" in roles
        assert "assistant" in roles
    finally:
        set_default_learning_store(None)
        set_default_progress_service(None)

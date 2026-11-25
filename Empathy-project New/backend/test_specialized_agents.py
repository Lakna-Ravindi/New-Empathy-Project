"""Unit tests for the four specialized agents in Empathy Guru.

Tests:
1. TutorAgent: Grounded 4-stage responses, fallback handling, RBAC boundaries.
2. EmotionalSupportAgent: Active listening, no-advice respect, non-clinical validation.
3. LearningPlannerAgent: Progress reading, canonical objective selection, read-only boundaries.
4. AssessmentAgent: Verification, authorized progress commitment, proof requirements.
"""

import pytest
from unittest.mock import MagicMock, patch

from agents.definitions import (
    AgentExecutionError,
    AssessmentAgent,
    AssessmentInput,
    EmotionalSupportAgent,
    EmotionalSupportInput,
    LearningPlannerAgent,
    LearningPlannerInput,
    TutorAgent,
    TutorInput,
)
from agents.state import (
    AgentState,
    InvalidEmotionalContextError,
    LearningStage,
)
from agents.tools import get_default_tool_registry
from agents.tools.registry import ToolPermissionError


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def tool_registry():
    return get_default_tool_registry()


@pytest.fixture
def mock_learning_store():
    store = MagicMock()
    store.get_all_progress.return_value = [
        {
            "student_id": "STU-001",
            "skill_id": "skill_01",
            "completed_item_ids": ["activity_01_01"],
            "completed_objective_ids": ["obj_01_01"],
            "quiz_scores": {},
        }
    ]
    store.get_progress.return_value = {
        "student_id": "STU-001",
        "skill_id": "skill_01",
        "completed_item_ids": ["activity_01_01"],
        "completed_objective_ids": ["obj_01_01"],
        "quiz_scores": {},
    }
    store.get_interaction.return_value = {
        "student_id": "STU-001",
        "question": "How do I practice grounding?",
        "learning_steps": [
            {"step": 1, "type": "direct_answer", "content": "Notice your feet."},
            {
                "step": 4,
                "type": "practice_check",
                "question": "True or False: Grounding calms the nervous system.",
                "correct_answer": True,
            },
        ],
    }
    return store


# =====================================================================
# TutorAgent Tests
# =====================================================================


def test_tutor_agent_execution_fallback(tool_registry):
    tutor = TutorAgent(tool_registry=tool_registry)

    tutor_input = TutorInput(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        student_question="How does somatic tracking help when I feel anxious?",
        skill_id="skill_01",
        objective_id="obj_01_01",
        answer_style="short",
    )

    state = AgentState(student_id="STU-001")
    output = tutor.execute(tutor_input, state=state)

    assert output.skill_id == "skill_01"
    assert output.objective_id == "obj_01_01"
    assert len(output.steps) == 4
    assert output.steps[0]["type"] == "direct_answer"
    assert output.steps[1]["type"] == "learning_objective"
    assert output.steps[2]["type"] == "activity"
    assert output.steps[3]["type"] == "practice_check"
    assert len(output.suggested_questions) >= 0
    assert state.current_learning_stage == LearningStage.TUTORING
    assert state.response is not None


def test_tutor_agent_identity_check(tool_registry):
    tutor = TutorAgent(tool_registry=tool_registry)
    tutor_input = TutorInput(
        student_id="STU-001",
        authenticated_student_id="STU-IMPOSTOR",
        student_question="Help me learn",
        skill_id="skill_01",
        objective_id="obj_01_01",
    )
    with pytest.raises(AgentExecutionError, match="Security violation"):
        tutor.execute(tutor_input)


def test_tutor_agent_cannot_call_update_student_progress(tool_registry):
    tutor = TutorAgent(tool_registry=tool_registry)
    # RBAC check: TutorAgent role cannot execute update_student_progress
    with pytest.raises(ToolPermissionError):
        tutor.call_tool(
            "update_student_progress",
            student_id="STU-001",
            authenticated_student_id="STU-001",
            authorizing_agent="tutor_agent",
            skill_id="skill_01",
        )


def test_tutor_agent_preserves_support_intro_fallback(tool_registry):
    tutor = TutorAgent(tool_registry=tool_registry)
    intro = "Exam stress can feel truly exhausting when expectations build up."
    tutor_input = TutorInput(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        student_question="How does somatic tracking help when I feel anxious?",
        skill_id="skill_01",
        objective_id="obj_01_01",
        answer_style="short",
        support_intro=intro,
    )
    output = tutor.execute(tutor_input)
    assert output.answer.startswith(intro)
    assert output.steps[0]["content"].startswith(intro)
    assert output.answer.count(intro) == 1
    assert output.steps[0]["content"].count(intro) == 1
    assert len(output.steps) == 4
    assert output.is_fallback is True


def test_tutor_agent_preserves_support_intro_gemini_success(tool_registry):
    import json
    tutor = TutorAgent(tool_registry=tool_registry)
    intro = "It takes courage to open up about how you're feeling."
    test_activity = {
        "id": "activity_07_01",
        "title": "Sympathetic Joy Cultivation",
        "type": "activity",
        "interaction_type": "practical",
    }
    tutor_input = TutorInput(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        student_question="What is empathy?",
        skill_id="skill_07",
        objective_id="obj_07_01",
        activity=test_activity,
        support_intro=intro,
    )

    mock_json_response = json.dumps({
        "answer": "Empathy is the capacity to understand and feel with others.",
        "suggested_questions": ["What is cognitive empathy?", "How do I practice empathy?", "Can empathy be learned?"],
        "steps": [
            {"step": 1, "type": "direct_answer", "content": "Empathy is the capacity to understand and feel with others."},
            {"step": 2, "type": "learning_objective", "content": "Understand empathic concern."},
            {"step": 3, "type": "activity", "content": "Practice active listening.", "activity": test_activity},
            {"step": 4, "type": "practice_check", "content": "Check your understanding.", "question_type": "true_false", "question": "Is empathy helpful?", "correct_answer": True},
        ]
    })

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = mock_json_response
    mock_client.models.generate_content.return_value = mock_resp

    with patch("os.getenv", side_effect=lambda k, default=None: "dummy_key" if k == "GEMINI_API_KEY" else default):
        with patch("agents.definitions.tutor_agent.genai.Client", return_value=mock_client):
            output = tutor.execute(tutor_input)

    assert output.is_fallback is False
    assert output.answer.startswith(intro)
    assert output.steps[0]["content"].startswith(intro)
    assert output.answer.count(intro) == 1
    assert output.steps[0]["content"].count(intro) == 1


def test_tutor_agent_does_not_duplicate_support_intro(tool_registry):
    import json
    tutor = TutorAgent(tool_registry=tool_registry)
    intro = "I hear you and I am here with you."
    test_activity = {
        "id": "activity_07_01",
        "title": "Sympathetic Joy Cultivation",
        "type": "activity",
        "interaction_type": "practical",
    }
    tutor_input = TutorInput(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        student_question="What is empathy?",
        skill_id="skill_07",
        objective_id="obj_07_01",
        activity=test_activity,
        support_intro=intro,
    )

    # Gemini already includes intro
    mock_json_response = json.dumps({
        "answer": f"{intro} Empathy is understanding others deeply.",
        "suggested_questions": ["What is cognitive empathy?", "How do I practice empathy?", "Can empathy be learned?"],
        "steps": [
            {"step": 1, "type": "direct_answer", "content": f"{intro} Empathy is understanding others deeply."},
            {"step": 2, "type": "learning_objective", "content": "Understand empathic concern."},
            {"step": 3, "type": "activity", "content": "Practice active listening.", "activity": test_activity},
            {"step": 4, "type": "practice_check", "content": "Check your understanding.", "question_type": "true_false", "question": "Is empathy helpful?", "correct_answer": True},
        ]
    })

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = mock_json_response
    mock_client.models.generate_content.return_value = mock_resp

    with patch("os.getenv", side_effect=lambda k, default=None: "dummy_key" if k == "GEMINI_API_KEY" else default):
        with patch("agents.definitions.tutor_agent.genai.Client", return_value=mock_client):
            output = tutor.execute(tutor_input)

    assert output.is_fallback is False
    assert output.answer.count(intro) == 1
    assert output.steps[0]["content"].count(intro) == 1


def test_tutor_agent_without_support_intro_untouched(tool_registry):
    tutor = TutorAgent(tool_registry=tool_registry)
    tutor_input = TutorInput(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        student_question="What is somatic tracking?",
        skill_id="skill_01",
        objective_id="obj_01_01",
        support_intro=None,
    )
    output = tutor.execute(tutor_input)
    assert len(output.steps) == 4
    assert output.steps[0]["type"] == "direct_answer"
    assert "Exam stress" not in output.answer
    assert "It takes courage" not in output.answer


# =====================================================================
# EmotionalSupportAgent Tests
# =====================================================================


def test_emotional_support_agent_basic(tool_registry):
    agent = EmotionalSupportAgent(tool_registry=tool_registry)
    user_input = EmotionalSupportInput(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        current_message="I'm feeling very overwhelmed with my exams and assignments.",
    )
    state = AgentState(student_id="STU-001")
    output = agent.execute(user_input, state=state)

    assert output.support_message != ""
    assert output.empathetic_acknowledgment != ""
    assert state.current_learning_stage == LearningStage.EMOTIONAL_SUPPORT
    assert state.emotional_context["is_clinical_diagnosis"] is False


def test_emotional_support_agent_respects_no_advice(tool_registry):
    agent = EmotionalSupportAgent(tool_registry=tool_registry)
    user_input = EmotionalSupportInput(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        current_message="Please don't give me advice, I just want to vent and have someone listen.",
    )
    output = agent.execute(user_input)

    assert output.emotional_context.get("no_advice_requested") is True
    assert "listen" in output.support_message.lower() or "hear you" in output.support_message.lower() or "here" in output.support_message.lower()


def test_emotional_support_agent_readiness_detection(tool_registry):
    agent = EmotionalSupportAgent(tool_registry=tool_registry)
    user_input = EmotionalSupportInput(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        current_message="I feel much calmer now. What should I do to practice this skill?",
    )
    output = agent.execute(user_input)

    assert output.readiness_for_learning is True
    assert output.gentle_transition_offer != ""


def test_emotional_support_agent_rejects_medical_diagnoses(tool_registry):
    agent = EmotionalSupportAgent(tool_registry=tool_registry)
    user_input = EmotionalSupportInput(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        current_message="I feel sad.",
        emotional_context={"clinical_diagnosis": "bipolar disorder"},
    )
    with pytest.raises(InvalidEmotionalContextError):
        agent.execute(user_input)


def test_emotional_support_agent_cannot_mutate_progress(tool_registry):
    agent = EmotionalSupportAgent(tool_registry=tool_registry)
    with pytest.raises(ToolPermissionError):
        agent.call_tool(
            "update_student_progress",
            student_id="STU-001",
            authenticated_student_id="STU-001",
            authorizing_agent="emotional_support_agent",
            skill_id="skill_01",
        )


# =====================================================================
# LearningPlannerAgent Tests
# =====================================================================


def test_learning_planner_agent_execution(tool_registry, mock_learning_store):
    planner = LearningPlannerAgent(tool_registry=tool_registry)

    planner_input = LearningPlannerInput(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        student_question="How can I calm down during a presentation?",
        current_skill_id="skill_01",
    )

    state = AgentState(student_id="STU-001")

    with patch("agents.tools.progress_tools.get_default_learning_store", return_value=mock_learning_store):
        output = planner.execute(planner_input, state=state)

    assert output.skill["id"] == "skill_01"
    assert output.learning_objective["id"] is not None
    assert isinstance(output.relevant_chunks, list)
    assert output.recommended_next_action in {"proceed_to_activity", "introduce_concept_and_check"}
    assert state.current_learning_stage == LearningStage.LEARNING_PLANNING
    assert state.current_skill_id == "skill_01"


def test_learning_planner_agent_cannot_mutate_progress(tool_registry):
    planner = LearningPlannerAgent(tool_registry=tool_registry)
    with pytest.raises(ToolPermissionError):
        planner.call_tool(
            "update_student_progress",
            student_id="STU-001",
            authenticated_student_id="STU-001",
            authorizing_agent="learning_planner_agent",
            skill_id="skill_01",
        )


# =====================================================================
# AssessmentAgent Tests
# =====================================================================


def test_assessment_agent_quick_check_correct(tool_registry, mock_learning_store):
    agent = AssessmentAgent(tool_registry=tool_registry)

    assessment_input = AssessmentInput(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        submission_type="quick_check",
        student_answer=True,
        objective_id="obj_01_02",
        skill_id="skill_01",
        item_id="activity_01_02",
        interaction_id="interaction-abc-123",
        commit_progress=True,
    )

    state = AgentState(student_id="STU-001", pending_objectives=["obj_01_02"])

    with patch("agents.tools.progress_tools.get_default_learning_store", return_value=mock_learning_store), \
         patch("agents.tools.evaluation_tools.get_default_learning_store", return_value=mock_learning_store):
        output = agent.execute(assessment_input, state=state)

    assert output.verified is True
    assert output.score == 100.0
    assert output.objective_completed is True
    assert output.progress_mutation_applied is True
    mock_learning_store.complete_item.assert_called_once()
    mock_learning_store.complete_objective.assert_called_once()
    assert "obj_01_02" in state.completed_objectives


def test_assessment_agent_quick_check_incorrect_no_progress_mutation(tool_registry, mock_learning_store):
    agent = AssessmentAgent(tool_registry=tool_registry)

    assessment_input = AssessmentInput(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        submission_type="quick_check",
        student_answer=False,  # Expected is True
        objective_id="obj_01_02",
        skill_id="skill_01",
        item_id="activity_01_02",
        interaction_id="interaction-abc-123",
        commit_progress=True,
    )

    state = AgentState(student_id="STU-001", pending_objectives=["obj_01_02"])

    with patch("agents.tools.progress_tools.get_default_learning_store", return_value=mock_learning_store), \
         patch("agents.tools.evaluation_tools.get_default_learning_store", return_value=mock_learning_store):
        output = agent.execute(assessment_input, state=state)

    assert output.verified is False
    assert output.objective_completed is False
    assert output.progress_mutation_applied is False
    # Never mutate progress when verification fails!
    mock_learning_store.complete_item.assert_not_called()
    mock_learning_store.complete_objective.assert_not_called()
    assert "obj_01_02" not in state.completed_objectives


def test_assessment_agent_reflection_evaluation(tool_registry, mock_learning_store):
    agent = AssessmentAgent(tool_registry=tool_registry)

    assessment_input = AssessmentInput(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        submission_type="reflection",
        student_answer="I noticed my heart rate calmed down when I focused on contact grounding.",
        objective_id="obj_01_02",
        skill_id="skill_01",
        item_id="activity_01_02",
        commit_progress=False,  # Just evaluation
    )

    with patch("agents.tools.progress_tools.get_default_learning_store", return_value=mock_learning_store), \
         patch("agents.tools.evaluation_tools.get_default_learning_store", return_value=mock_learning_store):
        output = agent.execute(assessment_input)

    assert output.verified is True
    assert output.score == 100.0
    assert output.progress_mutation_applied is False  # commit_progress was False


def test_assessment_agent_rejects_non_canonical_objective(tool_registry):
    agent = AssessmentAgent(tool_registry=tool_registry)
    assessment_input = AssessmentInput(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        submission_type="reflection",
        student_answer="test reflection",
        objective_id="obj_fake_99",
        skill_id="skill_01",
    )
    with pytest.raises(AgentExecutionError, match="not a canonical SEEK objective"):
        agent.execute(assessment_input)

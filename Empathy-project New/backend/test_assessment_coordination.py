"""Phase 4 – AssessmentAgent Coordination Integration Tests.

Validates:
1. Reflection submissions can evaluate/record responses but never complete objectives.
2. Quick checks require valid persisted interactions, matching skill/objective IDs,
   and strictly verified answers.
3. Quick checks cannot complete an objective if the required activity has not been completed.
4. Valid activity completion + verified quick check successfully completes the objective.
5. Quiz submissions rely on server-persisted quiz results (>= 90%), rejecting client-provided scores.
6. AssessmentAgent coordinates state transitions and records completions accurately.
7. Orchestrator delegation to AssessmentAgent preserves student isolation and contracts.
"""

from unittest.mock import MagicMock
import pytest

from agents.definitions import AssessmentAgent, AssessmentInput
from agents.orchestrator import EmpathyOrchestrator
from agents.state import AgentState, LearningStage
from agents.tools import get_default_tool_registry
from agents.tools.curriculum_tools import get_progress_structure
from agents.tools.progress_tools import (
    set_default_learning_store,
    set_default_progress_service,
)
from learning.progress_service import ProgressService


@pytest.fixture
def clean_tool_registry():
    return get_default_tool_registry()


@pytest.fixture
def mock_stores():
    store = MagicMock(spec=["get_progress", "get_all_progress", "get_interaction", "complete_item", "complete_objective"])
    service = ProgressService(structure=get_progress_structure())

    store.get_progress.return_value = {
        "student_id": "STU-001",
        "skill_id": "skill_01",
        "completed_item_ids": [],
        "completed_objective_ids": [],
        "quiz_scores": {},
    }
    store.get_all_progress.return_value = []

    # Default valid interaction for obj_01_01
    store.get_interaction.return_value = {
        "student_id": "STU-001",
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

    set_default_learning_store(store)
    set_default_progress_service(service)

    yield store, service

    set_default_learning_store(None)
    set_default_progress_service(None)


def test_reflection_submission_never_completes_objective(clean_tool_registry, mock_stores):
    """Reflections may evaluate thoughtfully, but must NEVER trigger objective completion."""
    store, service = mock_stores
    agent = AssessmentAgent(tool_registry=clean_tool_registry)

    input_data = AssessmentInput(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        submission_type="reflection",
        student_answer="I took five deep breaths and felt the physical tension leave my shoulders.",
        objective_id="obj_01_01",
        skill_id="skill_01",
        commit_progress=True,
    )

    output = agent.execute(input_data)

    assert output.verified is True
    assert output.score == 100.0
    assert output.objective_completed is False
    assert output.progress_mutation_applied is False
    store.complete_objective.assert_not_called()


def test_quick_check_incorrect_answer_rejected(clean_tool_registry, mock_stores):
    """Quick check with incorrect answer must fail verification and not complete objective."""
    store, service = mock_stores
    agent = AssessmentAgent(tool_registry=clean_tool_registry)

    input_data = AssessmentInput(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        submission_type="quick_check",
        student_answer=False,
        objective_id="obj_01_01",
        skill_id="skill_01",
        interaction_id="inter-001",
        commit_progress=True,
    )

    output = agent.execute(input_data)

    assert output.verified is False
    assert output.objective_completed is False
    assert output.progress_mutation_applied is False
    store.complete_objective.assert_not_called()


def test_quick_check_mismatched_skill_rejected(clean_tool_registry, mock_stores):
    """Interaction skill_id mismatching requested skill must be rejected."""
    store, service = mock_stores
    agent = AssessmentAgent(tool_registry=clean_tool_registry)

    input_data = AssessmentInput(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        submission_type="quick_check",
        student_answer=True,
        objective_id="obj_01_01",
        skill_id="skill_02",  # Mismatch!
        interaction_id="inter-001",
        commit_progress=True,
    )

    output = agent.execute(input_data)

    assert output.verified is False
    assert "does not match requested skill" in output.feedback
    assert output.objective_completed is False
    store.complete_objective.assert_not_called()


def test_quick_check_mismatched_objective_rejected(clean_tool_registry, mock_stores):
    """Interaction objective_id mismatching requested objective must be rejected."""
    store, service = mock_stores
    agent = AssessmentAgent(tool_registry=clean_tool_registry)

    input_data = AssessmentInput(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        submission_type="quick_check",
        student_answer=True,
        objective_id="obj_01_02",  # Mismatch!
        skill_id="skill_01",
        interaction_id="inter-001",
        commit_progress=True,
    )

    output = agent.execute(input_data)

    assert output.verified is False
    assert "does not match requested objective" in output.feedback
    assert output.objective_completed is False
    store.complete_objective.assert_not_called()


def test_quick_check_missing_activity_completion_rejected(clean_tool_registry, mock_stores):
    """Even if quick check is correct, objective cannot complete without required activity completed."""
    store, service = mock_stores
    agent = AssessmentAgent(tool_registry=clean_tool_registry)

    store.get_progress.return_value = {
        "student_id": "STU-001",
        "skill_id": "skill_01",
        "completed_item_ids": [],  # activity_01_01 is NOT completed
        "completed_objective_ids": [],
        "quiz_scores": {},
    }

    input_data = AssessmentInput(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        submission_type="quick_check",
        student_answer=True,
        objective_id="obj_01_01",
        skill_id="skill_01",
        interaction_id="inter-001",
        commit_progress=True,
    )

    output = agent.execute(input_data)

    assert output.verified is True
    assert output.objective_completed is False
    assert output.progress_mutation_applied is False
    assert "required activity has not been completed" in output.feedback
    store.complete_objective.assert_not_called()


def test_valid_activity_plus_verified_quick_check_completes_objective(clean_tool_registry, mock_stores):
    """When required activity is already completed and quick check is correct, objective completes."""
    store, service = mock_stores
    agent = AssessmentAgent(tool_registry=clean_tool_registry)

    progress_state = {
        "student_id": "STU-001",
        "skill_id": "skill_01",
        "completed_item_ids": ["activity_01_01"],
        "completed_objective_ids": [],
        "quiz_scores": {},
    }

    def mock_get_progress(*args, **kwargs):
        return dict(progress_state)

    def mock_complete_objective(*args, **kwargs):
        progress_state["completed_objective_ids"] = ["obj_01_01"]
        return dict(progress_state)

    store.get_progress.side_effect = mock_get_progress
    store.complete_objective.side_effect = mock_complete_objective

    state = AgentState(student_id="STU-001", pending_objectives=["obj_01_01"])

    input_data = AssessmentInput(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        submission_type="quick_check",
        student_answer=True,
        objective_id="obj_01_01",
        skill_id="skill_01",
        interaction_id="inter-001",
        commit_progress=True,
    )

    output = agent.execute(input_data, state=state)

    assert output.verified is True
    assert output.objective_completed is True
    assert output.progress_mutation_applied is True
    store.complete_objective.assert_called_once()
    call_args = store.complete_objective.call_args
    assert call_args[0][0] == "STU-001"
    assert call_args[0][1] == "skill_01"
    assert call_args[0][2]["id"] == "obj_01_01"
    assert call_args[0][3]["id"] == "obj_01_05"
    assert "obj_01_01" in state.completed_objectives
    assert state.current_learning_stage == LearningStage.ASSESSMENT


def test_forged_quiz_score_cannot_complete_objective(clean_tool_registry, mock_stores):
    """Client-provided quiz score is untrusted; missing or failing server record rejects completion."""
    store, service = mock_stores
    agent = AssessmentAgent(tool_registry=clean_tool_registry)

    store.get_progress.return_value = {
        "student_id": "STU-001",
        "skill_id": "skill_01",
        "completed_item_ids": ["activity_01_01"],
        "completed_objective_ids": [],
        "quiz_scores": {"quiz_01_01": 75.0},  # Stored server score is failing (< 90)
    }

    input_data = AssessmentInput(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        submission_type="quiz",
        student_answer=100.0,  # Client attempts to claim 100%
        objective_id="obj_01_01",
        skill_id="skill_01",
        commit_progress=True,
    )

    output = agent.execute(input_data)

    assert output.objective_completed is False
    assert output.progress_mutation_applied is False
    assert "quiz has not been passed in trusted server records" in output.feedback
    store.complete_objective.assert_not_called()


def test_orchestrator_handle_objective_completion_coordination(clean_tool_registry, mock_stores):
    """Orchestrator coordinates objective completion through AssessmentAgent seamlessly."""
    store, service = mock_stores

    progress_state = {
        "student_id": "STU-001",
        "skill_id": "skill_01",
        "completed_item_ids": ["activity_01_01"],
        "completed_objective_ids": [],
        "quiz_scores": {},
    }

    def mock_get_progress(*args, **kwargs):
        return dict(progress_state)

    def mock_complete_objective(*args, **kwargs):
        progress_state["completed_objective_ids"] = ["obj_01_01"]
        return dict(progress_state)

    store.get_progress.side_effect = mock_get_progress
    store.complete_objective.side_effect = mock_complete_objective

    orchestrator = EmpathyOrchestrator(
        learning_store=store,
        progress_service=service,
        tool_registry=clean_tool_registry,
    )

    output = orchestrator.handle_objective_completion(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        skill_id="skill_01",
        objective_id="obj_01_01",
        quick_check_answer=True,
        interaction_id="inter-001",
    )

    assert output.verified is True
    assert output.objective_completed is True
    store.complete_objective.assert_called_once()
    call_args = store.complete_objective.call_args
    assert call_args[0][0] == "STU-001"
    assert call_args[0][1] == "skill_01"
    assert call_args[0][2]["id"] == "obj_01_01"
    assert call_args[0][3]["id"] == "obj_01_05"

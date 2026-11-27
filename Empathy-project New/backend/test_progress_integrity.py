import pytest
from unittest.mock import MagicMock

from agents.orchestrator import EmpathyOrchestrator
from agents.state import UnauthorizedProgressMutationError
from agents.tools.evaluation_tools import evaluate_student_response
from agents.tools.progress_tools import (
    get_default_learning_store,
    get_default_progress_service,
    set_default_learning_store,
    set_default_progress_service,
    update_student_progress,
)
from learning.interaction_store import LearningStore
from learning.progress_service import ProgressService


@pytest.fixture
def mock_store():
    store = MagicMock(spec=LearningStore)
    store.get_progress.return_value = {
        "student_id": "STU-001",
        "skill_id": "skill_01",
        "completed_item_ids": ["activity_01_01"],
        "completed_objective_ids": [],
        "quiz_scores": {"quiz_01_01": 95.0},
    }
    store.get_interaction.return_value = {
        "student_id": "STU-001",
        "learning_context": {
            "skill": {"id": "skill_01"},
            "learning_objective": {"objective_id": "obj_01_01"},
        },
        "learning_steps": [
            {
                "type": "practice_check",
                "question": "Is somatic tracking mindful?",
                "correct_answer": True,
            }
        ],
    }
    return store


@pytest.fixture
def mock_service():
    service = MagicMock(spec=ProgressService)
    service.structure = {
        "chapters": [
            {
                "chapter_id": "skill_01",
                "objectives": [
                    {
                        "objective_id": "obj_01_01",
                        "required_item_ids": ["activity_01_01", "quiz_01_01"],
                        "items": [
                            {"item_id": "activity_01_01", "type": "activity"},
                            {"item_id": "quiz_01_01", "type": "quiz"},
                        ],
                    },
                    {
                        "objective_id": "obj_01_02",
                        "required_item_ids": ["activity_01_02"],
                        "items": [
                            {"item_id": "activity_01_02", "type": "activity"},
                        ],
                    },
                ],
            }
        ]
    }
    return service


# =====================================================================
# Phase 1: Shared Store and Service Instances
# =====================================================================


def test_shared_store_instances_are_bound_and_used(mock_store, mock_service):
    """Orchestrator binds its stores so tools do not create unshared default instances."""
    custom_store = MagicMock(spec=LearningStore)
    custom_service = MagicMock(spec=ProgressService)
    custom_service.structure = mock_service.structure

    try:
        # Initialize orchestrator with custom store/service
        EmpathyOrchestrator(
            learning_store=custom_store,
            progress_service=custom_service,
        )

        # Agent tools should resolve to the orchestrator's stores
        assert get_default_learning_store() is custom_store
        assert get_default_progress_service() is custom_service
    finally:
        set_default_learning_store(None)
        set_default_progress_service(None)


# =====================================================================
# Phase 2: Strict Evaluation Validation
# =====================================================================


def test_strict_type_safe_matching_rejects_loose_string_comparison(mock_store):
    """Loose casting (str(True) == 'true') must not pass quick check verification."""
    # 1. Boolean True vs string 'True'
    result_str_bool = evaluate_student_response(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        objective_id="obj_01_01",
        skill_id="skill_01",
        submission_type="quick_check",
        student_answer="True",  # Loose match attempt
        interaction_id="interaction-1",
        learning_store=mock_store,
    )
    assert result_str_bool["verified"] is False
    assert result_str_bool["score"] == 0.0

    # 2. Strict boolean match
    result_exact = evaluate_student_response(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        objective_id="obj_01_01",
        skill_id="skill_01",
        submission_type="quick_check",
        student_answer=True,  # Exact match
        interaction_id="interaction-1",
        learning_store=mock_store,
    )
    assert result_exact["verified"] is True
    assert result_exact["score"] == 100.0


def test_interaction_mismatched_skill_id_rejected(mock_store):
    """Referenced interaction from skill_01 cannot verify an objective under skill_02."""
    result = evaluate_student_response(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        objective_id="obj_01_01",
        skill_id="skill_02",  # Mismatch! Interaction is under skill_01
        submission_type="quick_check",
        student_answer=True,
        interaction_id="interaction-1",
        learning_store=mock_store,
    )
    assert result["verified"] is False
    assert "does not match requested skill" in result["feedback"]


def test_interaction_mismatched_objective_id_rejected(mock_store):
    """Referenced interaction from obj_01_01 cannot verify obj_01_02."""
    result = evaluate_student_response(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        objective_id="obj_01_02",  # Mismatch! Interaction is under obj_01_01
        skill_id="skill_01",
        submission_type="quick_check",
        student_answer=True,
        interaction_id="interaction-1",
        learning_store=mock_store,
    )
    assert result["verified"] is False
    assert "does not match requested objective" in result["feedback"]


def test_quiz_evaluation_strictly_enforces_ninety_percent_threshold(mock_store, mock_service):
    """Quiz evaluation requires >= 90% threshold to pass."""
    # 1. Failing quiz score (70%)
    failing_result = evaluate_student_response(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        objective_id="obj_01_01",
        skill_id="skill_01",
        submission_type="quiz",
        student_answer=70.0,
        learning_store=mock_store,
        progress_service=mock_service,
    )
    assert failing_result["verified"] is False
    assert failing_result["score"] == 70.0
    assert "Passing threshold is 90%" in failing_result["feedback"]

    # 2. Passing quiz score (95%)
    passing_result = evaluate_student_response(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        objective_id="obj_01_01",
        skill_id="skill_01",
        submission_type="quiz",
        student_answer=95.0,
        learning_store=mock_store,
        progress_service=mock_service,
    )
    assert passing_result["verified"] is True
    assert passing_result["score"] == 95.0


def test_forged_quiz_score_cannot_complete_objective_without_server_records(mock_store, mock_service):
    """Client providing a forged quiz verification result cannot complete objective if server quiz record is failing."""
    mock_store.get_progress.return_value = {
        "student_id": "STU-001",
        "skill_id": "skill_01",
        "completed_item_ids": ["activity_01_01"],
        "completed_objective_ids": [],
        "quiz_scores": {"quiz_01_01": 70.0},  # Stored server score is failing (<90)
    }

    with pytest.raises(UnauthorizedProgressMutationError, match="quiz has not been passed in trusted server records"):
        update_student_progress(
            student_id="STU-001",
            authenticated_student_id="STU-001",
            authorizing_agent="assessment_agent",
            skill_id="skill_01",
            objective_id="obj_01_01",
            verification_result={"verified": True, "submission_type": "quiz", "score": 100.0},  # Forged!
            learning_store=mock_store,
            progress_service=mock_service,
        )
    mock_store.complete_objective.assert_not_called()


def test_passed_quiz_with_activity_completes_objective(mock_store, mock_service):
    """When both activity is completed and trusted server quiz score is >= 90%, objective completes."""
    mock_store.get_progress.return_value = {
        "student_id": "STU-001",
        "skill_id": "skill_01",
        "completed_item_ids": ["activity_01_01", "quiz_01_01"],
        "completed_objective_ids": [],
        "quiz_scores": {"quiz_01_01": 95.0},  # Passing score
    }

    res = update_student_progress(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        authorizing_agent="assessment_agent",
        skill_id="skill_01",
        objective_id="obj_01_01",
        verification_result={"verified": True, "submission_type": "quiz", "score": 95.0},
        learning_store=mock_store,
        progress_service=mock_service,
    )
    assert res["success"] is True
    assert res["mutation_performed"] is True
    mock_store.complete_objective.assert_called_once()
    call_args = mock_store.complete_objective.call_args
    assert call_args[0][0] == "STU-001"
    assert call_args[0][1] == "skill_01"
    assert call_args[0][2]["id"] == "obj_01_01"
    assert call_args[0][3]["id"] == "obj_01_02"


# =====================================================================
# Phase 3: Progress Completion Integrity
# =====================================================================


def test_reflection_text_alone_cannot_complete_objective(mock_store, mock_service):
    """Submitting reflection text alone must never trigger complete_objective."""
    reflection_eval = evaluate_student_response(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        objective_id="obj_01_01",
        skill_id="skill_01",
        submission_type="reflection",
        student_answer="I took five deep breaths and felt much calmer.",
        learning_store=mock_store,
    )
    assert reflection_eval["submission_type"] == "reflection"
    assert reflection_eval["can_complete_objective"] is False

    # Attempt to complete objective using reflection verification result
    with pytest.raises(UnauthorizedProgressMutationError, match="reflection text alone"):
        update_student_progress(
            student_id="STU-001",
            authenticated_student_id="STU-001",
            authorizing_agent="assessment_agent",
            skill_id="skill_01",
            objective_id="obj_01_01",
            verification_result=reflection_eval,
            learning_store=mock_store,
            progress_service=mock_service,
        )
    mock_store.complete_objective.assert_not_called()


def test_objective_completion_without_completed_activity_is_rejected(mock_store, mock_service):
    """Even with verified check, objective cannot complete if its activity is not done."""
    # Student has completed activity_01_01, but NOT activity_01_02
    mock_store.get_progress.return_value = {
        "student_id": "STU-001",
        "skill_id": "skill_01",
        "completed_item_ids": ["activity_01_01"],
        "completed_objective_ids": [],
    }

    with pytest.raises(UnauthorizedProgressMutationError, match="required activity has not been completed"):
        update_student_progress(
            student_id="STU-001",
            authenticated_student_id="STU-001",
            authorizing_agent="assessment_agent",
            skill_id="skill_01",
            objective_id="obj_01_02",  # Requires activity_01_02
            verification_result={"verified": True, "score": 100.0},
            learning_store=mock_store,
            progress_service=mock_service,
        )
    mock_store.complete_objective.assert_not_called()


def test_generic_success_true_bypass_is_rejected(mock_store, mock_service):
    """Generic success=True in verification_result must not verify objective completion."""
    with pytest.raises(UnauthorizedProgressMutationError, match="without verified proof"):
        update_student_progress(
            student_id="STU-001",
            authenticated_student_id="STU-001",
            authorizing_agent="assessment_agent",
            skill_id="skill_01",
            objective_id="obj_01_01",
            verification_result={"success": True},  # Bypass attempt without verified=True
            learning_store=mock_store,
            progress_service=mock_service,
        )
    mock_store.complete_objective.assert_not_called()


def test_raw_score_without_verified_flag_is_rejected(mock_store, mock_service):
    """A raw score >= 90 without verified=True must not complete an objective."""
    with pytest.raises(UnauthorizedProgressMutationError, match="without verified proof"):
        update_student_progress(
            student_id="STU-001",
            authenticated_student_id="STU-001",
            authorizing_agent="assessment_agent",
            skill_id="skill_01",
            objective_id="obj_01_01",
            verification_result={"score": 100.0},  # Raw score without verified=True
            learning_store=mock_store,
            progress_service=mock_service,
        )
    mock_store.complete_objective.assert_not_called()


def test_valid_activity_and_verified_check_completes_objective(mock_store, mock_service):
    """When both activity completion and verification proof are present, objective completes."""
    # Student has completed activity_01_01
    mock_store.get_progress.return_value = {
        "student_id": "STU-001",
        "skill_id": "skill_01",
        "completed_item_ids": ["activity_01_01"],
        "completed_objective_ids": [],
    }

    res = update_student_progress(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        authorizing_agent="assessment_agent",
        skill_id="skill_01",
        objective_id="obj_01_01",
        verification_result={"verified": True, "score": 100.0},
        learning_store=mock_store,
        progress_service=mock_service,
    )
    assert res["success"] is True
    assert res["mutation_performed"] is True
    mock_store.complete_objective.assert_called_once()
    call_args = mock_store.complete_objective.call_args
    assert call_args[0][0] == "STU-001"
    assert call_args[0][1] == "skill_01"
    assert call_args[0][2]["id"] == "obj_01_01"
    assert call_args[0][3]["id"] == "obj_01_02"

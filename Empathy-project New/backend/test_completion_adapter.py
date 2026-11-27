"""Phase 5C: HTTP Completion Adapter Tests.

Validates that ObjectiveCompletionAdapter and orchestrator.complete_objective_http
translate AssessmentAgent evaluations and progress operations into the exact HTTP contracts
expected by the frontend, preserving:
1. Valid completion returns exact HTTP 200 progress document contract.
2. Missing activity returns exact HTTP 422 prerequisite failure contract.
3. Missing quiz returns exact HTTP 422 prerequisite failure contract.
4. Incorrect quick check returns HTTP 422 failure contract.
5. Reflection submission returns HTTP 422 (never completes objective).
6. Database failure returns HTTP 503 (PyMongoError handling).
7. Duplicate completion preserves existing progress (idempotent HTTP 200).
8. Unauthorized student returns HTTP 403.
9. Invalid input returns HTTP 400 and unauthenticated returns HTTP 401.
"""

from unittest.mock import MagicMock
import pytest
from pymongo.errors import PyMongoError

from agents.completion_adapter import ObjectiveCompletionAdapter
from agents.definitions import AssessmentAgent
from agents.orchestrator import EmpathyOrchestrator
from agents.tools import get_default_tool_registry
from agents.tools.curriculum_tools import get_progress_structure
from agents.tools.progress_tools import (
    set_default_learning_store,
    set_default_progress_service,
)
from learning.interaction_store import LearningStore
from learning.progress_service import ProgressService


@pytest.fixture
def clean_tool_registry():
    return get_default_tool_registry()


@pytest.fixture
def mock_store_and_service():
    """Provides isolated mock LearningStore and real ProgressService."""
    store = MagicMock(spec=["get_progress", "get_all_progress", "get_interaction", "complete_item", "complete_objective"])
    service = ProgressService(structure=get_progress_structure())

    store.get_progress.return_value = {
        "student_id": "STU-001",
        "skill_id": "skill_01",
        "completed_item_ids": [],
        "completed_objective_ids": [],
        "quiz_scores": {},
        "next_recommended_learning_objective": None,
        "updated_at": "2026-01-01T00:00:00Z",
    }
    store.get_all_progress.return_value = []

    # Valid interaction for obj_01_01
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


# 1. Valid completion returns exact HTTP 200 payload
def test_valid_completion_returns_exact_200_payload(clean_tool_registry, mock_store_and_service):
    """Activity completed + verified quick check returns HTTP 200 with full progress schema."""
    store, service = mock_store_and_service

    progress_state = {
        "student_id": "STU-001",
        "skill_id": "skill_01",
        "completed_item_ids": ["activity_01_01"],
        "completed_objective_ids": [],
        "quiz_scores": {},
        "next_recommended_learning_objective": None,
        "updated_at": "2026-01-01T00:00:00Z",
    }

    def mock_get_progress(student_id, skill_id):
        return dict(progress_state)

    def mock_complete_objective(student_id, skill_id, obj_rec, next_obj):
        progress_state["completed_objective_ids"] = ["obj_01_01"]
        progress_state["next_recommended_learning_objective"] = next_obj
        return dict(progress_state)

    store.get_progress.side_effect = mock_get_progress
    store.complete_objective.side_effect = mock_complete_objective

    adapter = ObjectiveCompletionAdapter(
        assessment_agent=AssessmentAgent(tool_registry=clean_tool_registry),
        learning_store=store,
        progress_service=service,
    )

    body = {
        "source": "chatbot",
        "interaction_id": "inter-001",
        "quick_check_answer": True,
    }

    payload, status = adapter.handle_completion_request(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        skill_id="skill_01",
        objective_id="obj_01_01",
        body=body,
    )

    assert status == 200
    assert payload["student_id"] == "STU-001"
    assert payload["skill_id"] == "skill_01"
    assert "obj_01_01" in payload["completed_objective_ids"]
    assert "activity_01_01" in payload["completed_item_ids"]
    assert payload["next_recommended_learning_objective"] is not None
    assert payload["next_recommended_learning_objective"]["id"] == "obj_01_05"


# 2. Missing activity returns exact HTTP 422 payload
def test_missing_activity_returns_exact_422_payload(clean_tool_registry, mock_store_and_service):
    """Correct quick check but missing required activity returns HTTP 422 with missing_item_ids."""
    store, service = mock_store_and_service

    store.get_progress.return_value = {
        "student_id": "STU-001",
        "skill_id": "skill_01",
        "completed_item_ids": [],
        "completed_objective_ids": [],
        "quiz_scores": {},
    }

    adapter = ObjectiveCompletionAdapter(
        assessment_agent=AssessmentAgent(tool_registry=clean_tool_registry),
        learning_store=store,
        progress_service=service,
    )

    body = {
        "source": "chatbot",
        "interaction_id": "inter-001",
        "quick_check_answer": True,
    }

    payload, status = adapter.handle_completion_request(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        skill_id="skill_01",
        objective_id="obj_01_01",
        body=body,
    )

    assert status == 422
    assert payload["completed"] is False
    assert payload["objective_id"] == "obj_01_01"
    assert "activity" in payload["missing_completion_conditions"]
    assert "activity_01_01" in payload["missing_item_ids"]
    assert "Complete the objective activity before completing this objective." in payload["error"]


# 3. Missing quiz returns exact HTTP 422 payload
def test_missing_quiz_returns_exact_422_payload(clean_tool_registry, mock_store_and_service):
    """Activity completed but quiz not passed returns HTTP 422 with missing quiz condition."""
    store, service = mock_store_and_service

    store.get_progress.return_value = {
        "student_id": "STU-001",
        "skill_id": "skill_01",
        "completed_item_ids": ["activity_01_01"],
        "completed_objective_ids": [],
        "quiz_scores": {"quiz_01_01": 70.0},  # Below 90% threshold
    }

    adapter = ObjectiveCompletionAdapter(
        assessment_agent=AssessmentAgent(tool_registry=clean_tool_registry),
        learning_store=store,
        progress_service=service,
    )

    body = {
        "source": "skills_page",
        "submission_type": "quiz",
        "quiz_score": 70.0,
    }

    payload, status = adapter.handle_completion_request(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        skill_id="skill_01",
        objective_id="obj_01_01",
        body=body,
    )

    assert status == 422
    assert payload["completed"] is False
    assert payload["objective_id"] == "obj_01_01"
    assert "quiz" in payload["missing_completion_conditions"]
    assert "activity_01_01" in payload["completed_item_ids"]
    assert "Complete the objective activity and pass its quiz" in payload["error"]


# 4. Incorrect quick check returns HTTP 422
def test_incorrect_quick_check_returns_422(clean_tool_registry, mock_store_and_service):
    """Incorrect quick check answer returns HTTP 422 requiring verified check."""
    store, service = mock_store_and_service

    store.get_progress.return_value = {
        "student_id": "STU-001",
        "skill_id": "skill_01",
        "completed_item_ids": ["activity_01_01"],
        "completed_objective_ids": [],
        "quiz_scores": {},
    }

    adapter = ObjectiveCompletionAdapter(
        assessment_agent=AssessmentAgent(tool_registry=clean_tool_registry),
        learning_store=store,
        progress_service=service,
    )

    body = {
        "source": "chatbot",
        "interaction_id": "inter-001",
        "quick_check_answer": False,  # Incorrect answer
    }

    payload, status = adapter.handle_completion_request(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        skill_id="skill_01",
        objective_id="obj_01_01",
        body=body,
    )

    assert status == 422
    assert payload["completed"] is False
    assert payload["objective_id"] == "obj_01_01"
    assert "A correct, verified understanding check is required" in payload["error"]


# 5. Reflection submission returns HTTP 422
def test_reflection_submission_returns_422(clean_tool_registry, mock_store_and_service):
    """Reflection submissions must NEVER complete objectives, returning HTTP 422."""
    store, service = mock_store_and_service

    store.get_progress.return_value = {
        "student_id": "STU-001",
        "skill_id": "skill_01",
        "completed_item_ids": ["activity_01_01"],
        "completed_objective_ids": [],
        "quiz_scores": {"quiz_01_01": 100.0},
    }

    adapter = ObjectiveCompletionAdapter(
        assessment_agent=AssessmentAgent(tool_registry=clean_tool_registry),
        learning_store=store,
        progress_service=service,
    )

    body = {
        "source": "skills_page",
        "submission_type": "reflection",
        "student_answer": "I reflected deeply on how noticing tension helps me pause.",
    }

    payload, status = adapter.handle_completion_request(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        skill_id="skill_01",
        objective_id="obj_01_01",
        body=body,
    )

    assert status == 422
    assert payload["completed"] is False
    assert "Objective cannot be marked complete from reflection text alone." in payload["error"]


# 6. Database failure returns HTTP 503
def test_database_failure_returns_503(clean_tool_registry, mock_store_and_service):
    """Storage PyMongoError returns canonical 503 service unavailable payload."""
    store, service = mock_store_and_service

    store.get_progress.side_effect = PyMongoError("Connection timed out to MongoDB cluster.")

    adapter = ObjectiveCompletionAdapter(
        assessment_agent=AssessmentAgent(tool_registry=clean_tool_registry),
        learning_store=store,
        progress_service=service,
    )

    payload, status = adapter.handle_completion_request(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        skill_id="skill_01",
        objective_id="obj_01_01",
        body={},
    )

    assert status == 503
    assert payload["error"] == "Could not update student progress."


# 7. Duplicate completion preserves existing progress (idempotent 200)
def test_duplicate_completion_is_idempotent_200(clean_tool_registry, mock_store_and_service):
    """If objective is already completed, returns existing progress with HTTP 200 without re-mutation."""
    store, service = mock_store_and_service

    existing_progress = {
        "student_id": "STU-001",
        "skill_id": "skill_01",
        "completed_item_ids": ["activity_01_01", "quiz_01_01"],
        "completed_objective_ids": ["obj_01_01"],
        "quiz_scores": {"quiz_01_01": 100.0},
        "next_recommended_learning_objective": {"id": "obj_01_05", "title": "Mindful Observation"},
        "updated_at": "2026-01-01T12:00:00Z",
    }
    store.get_progress.return_value = dict(existing_progress)

    adapter = ObjectiveCompletionAdapter(
        assessment_agent=AssessmentAgent(tool_registry=clean_tool_registry),
        learning_store=store,
        progress_service=service,
    )

    payload, status = adapter.handle_completion_request(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        skill_id="skill_01",
        objective_id="obj_01_01",
        body={"source": "skills_page"},
    )

    assert status == 200
    assert payload["completed_objective_ids"] == ["obj_01_01"]
    assert payload["next_recommended_learning_objective"]["id"] == "obj_01_05"
    store.complete_objective.assert_not_called()


# 8. Unauthorized student returns HTTP 403
def test_unauthorized_student_returns_403(clean_tool_registry, mock_store_and_service):
    """Mismatched student_id and authenticated_student_id returns HTTP 403."""
    store, service = mock_store_and_service

    adapter = ObjectiveCompletionAdapter(
        assessment_agent=AssessmentAgent(tool_registry=clean_tool_registry),
        learning_store=store,
        progress_service=service,
    )

    payload, status = adapter.handle_completion_request(
        student_id="STU-TARGET-002",
        authenticated_student_id="STU-ATTACKER-001",
        skill_id="skill_01",
        objective_id="obj_01_01",
        body={},
    )

    assert status == 403
    assert "Security violation" in payload["error"]


# 9. Invalid input returns HTTP 400 and unauthenticated returns HTTP 401
def test_invalid_input_and_unauthenticated(clean_tool_registry, mock_store_and_service):
    """Missing auth returns 401; invalid skill or objective IDs returns 400."""
    store, service = mock_store_and_service

    adapter = ObjectiveCompletionAdapter(
        assessment_agent=AssessmentAgent(tool_registry=clean_tool_registry),
        learning_store=store,
        progress_service=service,
    )

    # Missing auth token -> 401
    payload, status = adapter.handle_completion_request(
        student_id="STU-001",
        authenticated_student_id=None,
        skill_id="skill_01",
        objective_id="obj_01_01",
    )
    assert status == 401
    assert "Missing or invalid authorization" in payload["error"]

    # Non-canonical skill_id -> 400
    payload, status = adapter.handle_completion_request(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        skill_id="nonexistent_skill",
        objective_id="obj_01_01",
    )
    assert status == 400
    assert "valid skill_id and learning objective are required" in payload["error"]

    # Non-canonical objective_id -> 400
    payload, status = adapter.handle_completion_request(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        skill_id="skill_01",
        objective_id="nonexistent_objective",
    )
    assert status == 400
    assert "valid skill_id and learning objective are required" in payload["error"]


# 10. Orchestrator delegation verification
def test_orchestrator_complete_objective_http_delegation(clean_tool_registry, mock_store_and_service):
    """Orchestrator complete_objective_http routes through ObjectiveCompletionAdapter seamlessly."""
    store, service = mock_store_and_service

    progress_state = {
        "student_id": "STU-001",
        "skill_id": "skill_01",
        "completed_item_ids": ["activity_01_01"],
        "completed_objective_ids": [],
        "quiz_scores": {},
        "next_recommended_learning_objective": None,
        "updated_at": "2026-01-01T00:00:00Z",
    }

    def mock_get_progress(student_id, skill_id):
        return dict(progress_state)

    def mock_complete_objective(student_id, skill_id, obj_rec, next_obj):
        progress_state["completed_objective_ids"] = ["obj_01_01"]
        return dict(progress_state)

    store.get_progress.side_effect = mock_get_progress
    store.complete_objective.side_effect = mock_complete_objective

    orchestrator = EmpathyOrchestrator(
        learning_store=store,
        progress_service=service,
        tool_registry=clean_tool_registry,
    )

    payload, status = orchestrator.complete_objective_http(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        skill_id="skill_01",
        objective_id="obj_01_01",
        body={
            "source": "chatbot",
            "interaction_id": "inter-001",
            "quick_check_answer": True,
        },
    )

    assert status == 200
    assert "obj_01_01" in payload["completed_objective_ids"]

"""Phase 5B: LearningStore Compatibility Tests.

Validates that AssessmentAgent and update_student_progress interact seamlessly
with the real LearningStore implementation (including its canonical complete_objective
signature and dictionary parameters), verifying persistent storage integrity,
correct next-objective calculation, idempotency, and prerequisite enforcement.
"""

import pytest

from agents.definitions import AssessmentAgent, AssessmentInput
from agents.state import AgentState, UnauthorizedProgressMutationError
from agents.tools import get_default_tool_registry
from agents.tools.progress_tools import (
    set_default_learning_store,
    set_default_progress_service,
    update_student_progress,
)
from agents.tools.registry import ToolValidationError
from learning.interaction_store import LearningStore
from learning.progress_service import ProgressService


@pytest.fixture
def clean_tool_registry():
    return get_default_tool_registry()


@pytest.fixture
def real_store_and_service():
    """Provides a fresh real LearningStore (local storage mode) and ProgressService."""
    store = LearningStore()
    service = ProgressService()

    # Clear local storage progress for test isolation
    store.local_storage["progress"].clear()
    store.local_storage["interactions"].clear()

    set_default_learning_store(store)
    set_default_progress_service(service)

    yield store, service

    set_default_learning_store(None)
    set_default_progress_service(None)


def test_real_learning_store_valid_objective_completion(real_store_and_service):
    """Real LearningStore correctly records objective completion with canonical dict and next objective."""
    store, service = real_store_and_service
    student_id = "STU-REAL-001"
    skill_id = "skill_01"
    objective_id = "obj_01_01"

    # 1. Complete required activity first via tool
    item_res = update_student_progress(
        student_id=student_id,
        authenticated_student_id=student_id,
        authorizing_agent="assessment_agent",
        skill_id=skill_id,
        item_id="activity_01_01",
        item_type="activity",
        learning_store=store,
        progress_service=service,
    )
    assert item_res["success"] is True
    assert item_res["mutation_performed"] is True
    assert "activity_01_01" in item_res["completed_item_ids"]
    # Objective must NOT be marked complete yet (activity only)
    assert objective_id not in item_res["completed_objective_ids"]

    # 2. Complete objective via update_student_progress
    result = update_student_progress(
        student_id=student_id,
        authenticated_student_id=student_id,
        authorizing_agent="assessment_agent",
        skill_id=skill_id,
        objective_id=objective_id,
        verification_result={"verified": True, "submission_type": "quick_check", "score": 100.0},
        learning_store=store,
        progress_service=service,
    )

    assert result["success"] is True
    assert result["mutation_performed"] is True
    assert objective_id in result["completed_objective_ids"]

    # 3. Verify directly from persistent LearningStore
    persisted = store.get_progress(student_id, skill_id)
    assert objective_id in persisted["completed_objective_ids"]

    # In SEEK curriculum skill_01, obj_01_05 immediately follows obj_01_01
    next_obj = persisted.get("next_recommended_learning_objective")
    assert next_obj is not None
    assert next_obj["id"] == "obj_01_05"
    assert next_obj.get("title") != ""


def test_real_learning_store_existing_completed_objectives_remain_intact(real_store_and_service):
    """Subsequent objective completion preserves existing completed objectives and updates next."""
    store, service = real_store_and_service
    student_id = "STU-REAL-002"
    skill_id = "skill_01"

    # Pre-populate obj_01_01
    store.complete_objective(
        student_id=student_id,
        skill_id=skill_id,
        objective={"id": "obj_01_01", "title": "Mindful Breathing", "content": ""},
        next_objective={"id": "obj_01_05", "title": "Body Awareness", "content": ""},
    )
    # Complete activity for obj_01_02
    update_student_progress(
        student_id=student_id,
        authenticated_student_id=student_id,
        authorizing_agent="assessment_agent",
        skill_id=skill_id,
        item_id="activity_01_02",
        item_type="activity",
        learning_store=store,
        progress_service=service,
    )

    # Now complete obj_01_02
    result = update_student_progress(
        student_id=student_id,
        authenticated_student_id=student_id,
        authorizing_agent="assessment_agent",
        skill_id=skill_id,
        objective_id="obj_01_02",
        verification_result={"verified": True, "submission_type": "quick_check", "score": 100.0},
        learning_store=store,
        progress_service=service,
    )

    assert result["success"] is True
    persisted = store.get_progress(student_id, skill_id)
    # Both objectives must be present
    assert "obj_01_01" in persisted["completed_objective_ids"]
    assert "obj_01_02" in persisted["completed_objective_ids"]

    # Next uncompleted objective should be obj_01_05
    next_obj = persisted.get("next_recommended_learning_objective")
    assert next_obj is not None
    assert next_obj["id"] == "obj_01_05"


def test_real_learning_store_duplicate_completion_is_idempotent(real_store_and_service):
    """Attempting duplicate objective completion is idempotent and does not corrupt store."""
    store, service = real_store_and_service
    student_id = "STU-REAL-003"
    skill_id = "skill_01"
    objective_id = "obj_01_01"

    update_student_progress(
        student_id=student_id,
        authenticated_student_id=student_id,
        authorizing_agent="assessment_agent",
        skill_id=skill_id,
        item_id="activity_01_01",
        item_type="activity",
        learning_store=store,
        progress_service=service,
    )

    # First completion
    res1 = update_student_progress(
        student_id=student_id,
        authenticated_student_id=student_id,
        authorizing_agent="assessment_agent",
        skill_id=skill_id,
        objective_id=objective_id,
        verification_result={"verified": True, "submission_type": "quick_check", "score": 100.0},
        learning_store=store,
        progress_service=service,
    )
    assert res1["mutation_performed"] is True

    # Second identical completion
    res2 = update_student_progress(
        student_id=student_id,
        authenticated_student_id=student_id,
        authorizing_agent="assessment_agent",
        skill_id=skill_id,
        objective_id=objective_id,
        verification_result={"verified": True, "submission_type": "quick_check", "score": 100.0},
        learning_store=store,
        progress_service=service,
    )
    assert res2["success"] is True
    assert res2["objective_already_completed"] is True
    assert res2["mutation_performed"] is False

    # Stored list should have no duplicate items
    persisted = store.get_progress(student_id, skill_id)
    completed_ids = persisted["completed_objective_ids"]
    assert completed_ids.count(objective_id) == 1


def test_real_learning_store_invalid_objective_cannot_be_completed(real_store_and_service):
    """Non-canonical objective IDs fail validation and do not mutate persistent progress."""
    store, service = real_store_and_service
    student_id = "STU-REAL-004"
    skill_id = "skill_01"

    with pytest.raises(ToolValidationError, match="not a canonical SEEK objective"):
        update_student_progress(
            student_id=student_id,
            authenticated_student_id=student_id,
            authorizing_agent="assessment_agent",
            skill_id=skill_id,
            objective_id="obj_unknown_99",
            verification_result={"verified": True, "submission_type": "quick_check", "score": 100.0},
            learning_store=store,
            progress_service=service,
        )

    persisted = store.get_progress(student_id, skill_id)
    assert persisted.get("completed_objective_ids", []) == []


def test_real_learning_store_reflection_cannot_complete_objective(real_store_and_service):
    """Reflection submissions raise error and never complete an objective in persistent storage."""
    store, service = real_store_and_service
    student_id = "STU-REAL-005"
    skill_id = "skill_01"
    objective_id = "obj_01_01"

    update_student_progress(
        student_id=student_id,
        authenticated_student_id=student_id,
        authorizing_agent="assessment_agent",
        skill_id=skill_id,
        item_id="activity_01_01",
        item_type="activity",
        learning_store=store,
        progress_service=service,
    )

    with pytest.raises(UnauthorizedProgressMutationError, match="cannot be marked complete from reflection text alone"):
        update_student_progress(
            student_id=student_id,
            authenticated_student_id=student_id,
            authorizing_agent="assessment_agent",
            skill_id=skill_id,
            objective_id=objective_id,
            verification_result={"verified": True, "submission_type": "reflection", "score": 100.0},
            learning_store=store,
            progress_service=service,
        )

    persisted = store.get_progress(student_id, skill_id)
    assert objective_id not in persisted.get("completed_objective_ids", [])


def test_real_learning_store_failed_activity_prerequisite_does_not_mutate_progress(real_store_and_service):
    """Missing activity completion rejects objective completion and leaves storage unmodified."""
    store, service = real_store_and_service
    student_id = "STU-REAL-006"
    skill_id = "skill_01"
    objective_id = "obj_01_01"

    # Activity NOT completed in store
    with pytest.raises(UnauthorizedProgressMutationError, match="required activity has not been completed"):
        update_student_progress(
            student_id=student_id,
            authenticated_student_id=student_id,
            authorizing_agent="assessment_agent",
            skill_id=skill_id,
            objective_id=objective_id,
            verification_result={"verified": True, "submission_type": "quick_check", "score": 100.0},
            learning_store=store,
            progress_service=service,
        )

    persisted = store.get_progress(student_id, skill_id)
    assert objective_id not in persisted.get("completed_objective_ids", [])


def test_assessment_agent_with_real_learning_store_end_to_end(clean_tool_registry, real_store_and_service):
    """AssessmentAgent coordinates end-to-end evaluation and progress commitment with real LearningStore."""
    store, service = real_store_and_service
    student_id = "STU-REAL-007"
    skill_id = "skill_01"
    objective_id = "obj_01_01"

    # Complete activity item first via tool
    update_student_progress(
        student_id=student_id,
        authenticated_student_id=student_id,
        authorizing_agent="assessment_agent",
        skill_id=skill_id,
        item_id="activity_01_01",
        item_type="activity",
        learning_store=store,
        progress_service=service,
    )

    # Save interaction with practice check in the real store
    interaction_id = store.save_interaction(
        student_id=student_id,
        learning_context={
            "student_question": "How do I practice deep breathing?",
            "status": "learning_path_selected",
            "skill": {"id": skill_id},
            "learning_objective": {"id": objective_id},
        },
        educational_response="Mindful breathing explanation",
        learning_steps=[
            {
                "type": "practice_check",
                "question": "Is deep breathing calming?",
                "correct_answer": True,
            }
        ],
    )

    agent = AssessmentAgent(tool_registry=clean_tool_registry)
    state = AgentState(student_id=student_id, pending_objectives=[objective_id])

    input_data = AssessmentInput(
        student_id=student_id,
        authenticated_student_id=student_id,
        submission_type="quick_check",
        student_answer=True,
        objective_id=objective_id,
        skill_id=skill_id,
        interaction_id=interaction_id,
        commit_progress=True,
    )

    output = agent.execute(input_data, state=state)

    assert output.verified is True
    assert output.objective_completed is True
    assert output.progress_mutation_applied is True
    assert objective_id in state.completed_objectives

    # Verify directly from real LearningStore
    persisted = store.get_progress(student_id, skill_id)
    assert objective_id in persisted["completed_objective_ids"]
    assert persisted["next_recommended_learning_objective"]["id"] == "obj_01_05"

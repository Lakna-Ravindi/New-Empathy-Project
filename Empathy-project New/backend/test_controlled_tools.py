"""Comprehensive unit tests for Controlled Tool Execution Layer."""

import pytest
from unittest.mock import MagicMock

from agents.state import (
    UnauthorizedAccessError,
    UnauthorizedProgressMutationError,
)
from agents.tools import (
    ToolError,
    ToolExecutionError,
    ToolPermissionError,
    ToolRegistry,
    ToolValidationError,
    evaluate_student_response,
    get_current_skill,
    get_default_tool_registry,
    get_learning_activity,
    get_pending_objectives,
    get_student_progress,
    retrieve_approved_seek_content,
    save_learning_interaction,
    update_student_progress,
    validate_activity_completion,
)


# =====================================================================
# Mock Fixtures
# =====================================================================


@pytest.fixture
def mock_learning_store():
    store = MagicMock()
    # Default progress doc
    store.get_all_progress.return_value = [
        {
            "student_id": "STU-001",
            "skill_id": "skill_01",
            "completed_item_ids": ["activity_01_01"],
            "completed_objective_ids": ["obj_01_01"],
            "quiz_scores": {"quiz_01_01": 95},
        }
    ]
    store.get_progress.return_value = {
        "student_id": "STU-001",
        "skill_id": "skill_01",
        "completed_item_ids": ["activity_01_01"],
        "completed_objective_ids": ["obj_01_01"],
        "quiz_scores": {"quiz_01_01": 95},
    }
    store.save_interaction.return_value = "interaction-abc-123"
    store.get_interaction.return_value = {
        "student_id": "STU-001",
        "question": "How do I ground myself?",
        "learning_steps": [
            {"step": 1, "type": "direct_answer", "content": "Notice your feet on the floor."},
            {
                "step": 4,
                "type": "practice_check",
                "question": "True or False: Grounding helps with dysregulation.",
                "correct_answer": True,
            },
        ],
    }
    return store


@pytest.fixture
def mock_progress_service():
    service = MagicMock()
    service.structure = {
        "chapters": [
            {
                "chapter_id": "skill_01",
                "title": "Calming the Body and Mind",
                "objectives": [
                    {
                        "objective_id": "obj_01_01",
                        "title": "Somatic Tracking",
                        "required_item_ids": ["activity_01_01"],
                        "items": [
                            {"item_id": "activity_01_01", "type": "activity", "title": "Tracking Practice"},
                            {"item_id": "quiz_01_01", "type": "quiz", "title": "Tracking Quiz"},
                        ],
                    },
                    {
                        "objective_id": "obj_01_02",
                        "title": "Grounding Practice",
                        "required_item_ids": ["activity_01_02"],
                        "items": [
                            {"item_id": "activity_01_02", "type": "activity", "title": "Contact Grounding"},
                        ],
                    },
                ],
            },
            {
                "chapter_id": "skill_02",
                "title": "Ethical Mindfulness",
                "objectives": [
                    {
                        "objective_id": "obj_02_01",
                        "title": "Mindful Attention",
                        "required_item_ids": ["activity_02_01"],
                        "items": [
                            {"item_id": "activity_02_01", "type": "activity", "title": "Mindful Breathing"},
                        ],
                    }
                ],
            },
        ]
    }
    service.calculate.return_value = {
        "overall_completed_percentage": 25,
        "total_objectives_completed": 1,
        "skills": [
            {
                "skill_id": "skill_01",
                "title": "Calming the Body and Mind",
                "completed": False,
                "progress_percentage": 50,
                "completed_objective_ids": ["obj_01_01"],
                "objectives": [
                    {"objective_id": "obj_01_01", "completed": True},
                    {"objective_id": "obj_01_02", "completed": False},
                ],
            },
            {
                "skill_id": "skill_02",
                "title": "Ethical Mindfulness",
                "completed": False,
                "progress_percentage": 0,
                "completed_objective_ids": [],
                "objectives": [
                    {"objective_id": "obj_02_01", "completed": False},
                ],
            },
        ],
    }
    return service


# =====================================================================
# Tool 1: retrieve_approved_seek_content Tests
# =====================================================================


def test_retrieve_approved_seek_content_valid():
    results = retrieve_approved_seek_content(
        query="tracking somatic sensations in the body",
        skill_id="skill_01",
        limit=3,
    )
    assert isinstance(results, list)
    assert len(results) <= 3
    if results:
        first = results[0]
        assert "title" in first
        assert "content" in first
        assert "score" in first
        assert first["skill_id"] == "skill_01"


def test_retrieve_approved_seek_content_validation():
    with pytest.raises(ToolValidationError, match="query must be a non-empty string"):
        retrieve_approved_seek_content(query="")

    with pytest.raises(ToolValidationError, match="Invalid skill_id"):
        retrieve_approved_seek_content(query="test", skill_id="skill_fake_99")

    with pytest.raises(ToolValidationError, match="Invalid objective_id"):
        retrieve_approved_seek_content(query="test", objective_id="obj_fake_99")

    with pytest.raises(ToolValidationError, match="limit must be an integer between 1 and 20"):
        retrieve_approved_seek_content(query="test", limit=50)


# =====================================================================
# Tool 2: get_student_progress Tests
# =====================================================================


def test_get_student_progress_identity_verification(mock_learning_store, mock_progress_service):
    # Cross-student attempt must raise UnauthorizedAccessError
    with pytest.raises(UnauthorizedAccessError, match="Security violation"):
        get_student_progress(
            student_id="STU-001",
            authenticated_student_id="STU-002",
            learning_store=mock_learning_store,
            progress_service=mock_progress_service,
        )


def test_get_student_progress_read_only(mock_learning_store, mock_progress_service):
    result = get_student_progress(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        learning_store=mock_learning_store,
        progress_service=mock_progress_service,
    )
    assert result["student_id"] == "STU-001"
    assert result["overall_completed_percentage"] == 25
    assert len(result["skills"]) == 2
    # Ensure no mutation methods were called
    mock_learning_store.complete_item.assert_not_called()
    mock_learning_store.complete_objective.assert_not_called()


def test_get_student_progress_single_skill(mock_learning_store, mock_progress_service):
    result = get_student_progress(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        skill_id="skill_01",
        learning_store=mock_learning_store,
        progress_service=mock_progress_service,
    )
    assert result["student_id"] == "STU-001"
    assert result["skill"]["skill_id"] == "skill_01"
    assert result["skill"]["progress_percentage"] == 50


# =====================================================================
# Tool 3: get_pending_objectives Tests
# =====================================================================


def test_get_pending_objectives(mock_learning_store, mock_progress_service):
    pending = get_pending_objectives(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        skill_id="skill_01",
        learning_store=mock_learning_store,
        progress_service=mock_progress_service,
    )
    # obj_01_01 is completed, so only obj_01_02 should be pending
    assert len(pending) == 1
    assert pending[0]["objective_id"] == "obj_01_02"
    assert pending[0]["skill_id"] == "skill_01"
    assert pending[0]["is_completed"] is False


def test_get_pending_objectives_identity_check(mock_learning_store):
    with pytest.raises(UnauthorizedAccessError):
        get_pending_objectives(
            student_id="STU-001",
            authenticated_student_id="STU-OTHER",
            learning_store=mock_learning_store,
        )


# =====================================================================
# Tool 4: get_current_skill Tests
# =====================================================================


def test_get_current_skill(mock_learning_store, mock_progress_service):
    current = get_current_skill(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        learning_store=mock_learning_store,
        progress_service=mock_progress_service,
    )
    assert current["skill_id"] == "skill_01"
    assert current["completed"] is False
    assert current["next_objective_id"] == "obj_01_02"


# =====================================================================
# Tool 5: get_learning_activity Tests
# =====================================================================


def test_get_learning_activity_valid(mock_progress_service):
    activity = get_learning_activity(
        skill_id="skill_01",
        objective_id="obj_01_01",
        progress_structure=mock_progress_service.structure,
    )
    assert activity["id"] == "activity_01_01"
    assert activity["progress_item_id"] == "activity_01_01"
    assert activity["progress_item_type"] == "activity"
    assert activity["skill_id"] == "skill_01"
    assert activity["objective_id"] == "obj_01_01"


def test_get_learning_activity_rejects_hallucinations(mock_progress_service):
    with pytest.raises(ToolValidationError, match="Invalid skill_id"):
        get_learning_activity(skill_id="fake_skill", progress_structure=mock_progress_service.structure)

    with pytest.raises(ToolValidationError, match="not an approved activity"):
        get_learning_activity(
            skill_id="skill_01",
            activity_id="invented_activity_99",
            progress_structure=mock_progress_service.structure,
        )


# =====================================================================
# Tool 6: evaluate_student_response Tests
# =====================================================================


def test_evaluate_student_response_quick_check_correct(mock_learning_store):
    result = evaluate_student_response(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        objective_id="obj_01_01",
        submission_type="quick_check",
        student_answer=True,
        interaction_id="interaction-abc-123",
        learning_store=mock_learning_store,
    )
    assert result["verified"] is True
    assert result["correct"] is True
    assert result["score"] == 100.0


def test_evaluate_student_response_quick_check_incorrect(mock_learning_store):
    result = evaluate_student_response(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        objective_id="obj_01_01",
        submission_type="quick_check",
        student_answer=False,
        interaction_id="interaction-abc-123",
        learning_store=mock_learning_store,
    )
    assert result["verified"] is False
    assert result["correct"] is False
    assert result["score"] == 0.0


def test_evaluate_student_response_reflection(mock_learning_store):
    good_reflection = evaluate_student_response(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        objective_id="obj_01_01",
        submission_type="reflection",
        student_answer="I noticed tingling in my hands and my breath slowed down.",
        learning_store=mock_learning_store,
    )
    assert good_reflection["verified"] is True
    assert good_reflection["score"] == 100.0

    short_reflection = evaluate_student_response(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        objective_id="obj_01_01",
        submission_type="reflection",
        student_answer="ok",
        learning_store=mock_learning_store,
    )
    assert short_reflection["verified"] is False


def test_evaluate_student_response_identity_check(mock_learning_store):
    with pytest.raises(UnauthorizedAccessError):
        evaluate_student_response(
            student_id="STU-001",
            authenticated_student_id="STU-IMPOSTOR",
            objective_id="obj_01_01",
            submission_type="reflection",
            student_answer="Thinking about empathy",
            learning_store=mock_learning_store,
        )


# =====================================================================
# Tool 7: validate_activity_completion Tests
# =====================================================================


def test_validate_activity_completion_valid(mock_learning_store, mock_progress_service):
    res = validate_activity_completion(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        skill_id="skill_01",
        progress_item_id="activity_01_01",
        progress_item_type="activity",
        learning_store=mock_learning_store,
        progress_service=mock_progress_service,
    )
    assert res["is_valid"] is True
    assert res["skill_id"] == "skill_01"
    assert res["item_id"] == "activity_01_01"
    assert res["already_completed"] is True  # Already completed in mock


def test_validate_activity_completion_mismatched_type(mock_learning_store, mock_progress_service):
    res = validate_activity_completion(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        skill_id="skill_01",
        progress_item_id="activity_01_01",
        progress_item_type="quiz",  # Mismatch: item is activity
        learning_store=mock_learning_store,
        progress_service=mock_progress_service,
    )
    assert res["is_valid"] is False
    assert "item_type mismatch" in res["error"]


def test_validate_activity_completion_nonexistent_item(mock_learning_store, mock_progress_service):
    res = validate_activity_completion(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        skill_id="skill_01",
        progress_item_id="fake_item_999",
        progress_item_type="activity",
        learning_store=mock_learning_store,
        progress_service=mock_progress_service,
    )
    assert res["is_valid"] is False
    assert "not assigned" in res["error"]


# =====================================================================
# Tool 8: save_learning_interaction Tests
# =====================================================================


def test_save_learning_interaction(mock_learning_store):
    interaction_id = save_learning_interaction(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        question="What is tracking?",
        response_payload={"answer": "Tracking is noticing body sensations."},
        learning_context={
            "student_question": "What is tracking?",
            "status": "learning_path_selected",
            "skill": {"id": "skill_01"},
            "recommended_activity": {"progress_item_id": "activity_01_01", "progress_objective_id": "obj_01_01"},
        },
        learning_store=mock_learning_store,
    )
    assert interaction_id == "interaction-abc-123"
    mock_learning_store.save_interaction.assert_called_once()
    mock_learning_store.record_learning_event.assert_called_once_with(
        student_id="STU-001",
        skill_id="skill_01",
        item_id="activity_01_01",
        objective_id="obj_01_01",
        source="chatbot",
        status="introduced",
    )


def test_save_learning_interaction_identity_check(mock_learning_store):
    with pytest.raises(UnauthorizedAccessError):
        save_learning_interaction(
            student_id="STU-001",
            authenticated_student_id="STU-ATTACKER",
            question="What is tracking?",
            response_payload={"answer": "..."},
            learning_context={},
            learning_store=mock_learning_store,
        )


# =====================================================================
# Tool 9: update_student_progress Tests
# =====================================================================


def test_update_student_progress_authorized_agent(mock_learning_store, mock_progress_service):
    res = update_student_progress(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        authorizing_agent="assessment_agent",
        skill_id="skill_01",
        item_id="activity_01_02",
        objective_id="obj_01_02",
        item_type="activity",
        verification_result={"verified": True, "score": 100},
        learning_store=mock_learning_store,
        progress_service=mock_progress_service,
    )
    assert res["success"] is True
    assert res["mutation_performed"] is True
    mock_learning_store.complete_item.assert_called_once()
    mock_learning_store.complete_objective.assert_called_once()


def test_update_student_progress_unauthorized_agent_rejected(mock_learning_store, mock_progress_service):
    with pytest.raises(UnauthorizedProgressMutationError, match="not authorized to update progress"):
        update_student_progress(
            student_id="STU-001",
            authenticated_student_id="STU-001",
            authorizing_agent="tutor_agent",  # Tutor cannot mutate progress!
            skill_id="skill_01",
            item_id="activity_01_02",
            learning_store=mock_learning_store,
            progress_service=mock_progress_service,
        )


def test_update_student_progress_duplicate_prevention(mock_learning_store, mock_progress_service):
    # activity_01_01 and obj_01_01 are already completed in mock
    res = update_student_progress(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        authorizing_agent="assessment_agent",
        skill_id="skill_01",
        item_id="activity_01_01",
        objective_id="obj_01_01",
        item_type="activity",
        learning_store=mock_learning_store,
        progress_service=mock_progress_service,
    )
    assert res["item_already_completed"] is True
    assert res["objective_already_completed"] is True
    # DB complete calls should NOT be made for already completed items/objectives
    mock_learning_store.complete_item.assert_not_called()
    mock_learning_store.complete_objective.assert_not_called()


def test_update_student_progress_identity_check(mock_learning_store, mock_progress_service):
    with pytest.raises(UnauthorizedAccessError):
        update_student_progress(
            student_id="STU-001",
            authenticated_student_id="STU-VICTIM",
            authorizing_agent="assessment_agent",
            skill_id="skill_01",
            learning_store=mock_learning_store,
            progress_service=mock_progress_service,
        )


# =====================================================================
# ToolRegistry RBAC Tests
# =====================================================================


def test_tool_registry_rbac_enforcement(mock_learning_store, mock_progress_service):
    registry = get_default_tool_registry()

    # 1. Planner executing retrieve_approved_seek_content -> permitted
    result = registry.execute(
        agent_role="planner",
        tool_name="retrieve_approved_seek_content",
        query="empathy calming",
        limit=2,
    )
    assert isinstance(result, list)

    # 2. Tutor trying to execute update_student_progress -> forbidden!
    with pytest.raises(ToolPermissionError, match="not authorized to execute tool"):
        registry.execute(
            agent_role="tutor",
            tool_name="update_student_progress",
            student_id="STU-001",
            authenticated_student_id="STU-001",
            authorizing_agent="tutor",
            skill_id="skill_01",
        )

    # 3. Assessment agent executing update_student_progress -> permitted
    res = registry.execute(
        agent_role="assessment_agent",
        tool_name="update_student_progress",
        student_id="STU-001",
        authenticated_student_id="STU-001",
        authorizing_agent="assessment_agent",
        skill_id="skill_01",
        item_id="activity_01_01",
        learning_store=mock_learning_store,
        progress_service=mock_progress_service,
    )
    assert res["success"] is True

    # 4. Unknown tool -> validation error
    with pytest.raises(ToolValidationError, match="Unknown or unregistered tool"):
        registry.execute(agent_role="assessment", tool_name="arbitrary_mongo_query")

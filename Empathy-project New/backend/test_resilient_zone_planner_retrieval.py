"""Regression tests for Learning Planner Agent and SEEK Content Retrieval Integration.

Covers AUDIT-CONFIRMED ISSUES:
1. Resilient Zone query maps to the correct relevant objective (obj_01_02 / obj_01_06, not obj_01_01).
2. Retrieval does not discard all relevant chunks because of an unrelated pending objective.
3. Student progress is not modified by ordinary topic questions.
4. Unmatched questions preserve existing progress-based behavior.
5. Unapproved content is never returned.
"""

from unittest.mock import MagicMock, patch
import pytest

from agents.definitions import LearningPlannerAgent, LearningPlannerInput, TutorAgent, TutorInput
from agents.orchestrator import EmpathyOrchestrator
from agents.state import AgentState, LearningStage
from agents.tools import get_default_tool_registry
from agents.tools.curriculum_tools import retrieve_approved_seek_content, get_knowledge_base
from agents.tools.registry import ToolValidationError


@pytest.fixture
def tool_registry():
    return get_default_tool_registry()


@pytest.fixture
def mock_learning_store():
    store = MagicMock()
    store.get_progress.return_value = {
        "student_id": "STU-001",
        "skill_id": "skill_01",
        "completed_item_ids": [],
        "completed_objective_ids": [],
        "learning_events": [],
    }
    store.get_all_progress.return_value = [
        {
            "student_id": "STU-001",
            "skill_id": "skill_01",
            "completed_item_ids": [],
            "completed_objective_ids": [],
            "learning_events": [],
        }
    ]
    store.save_interaction.return_value = "mock_interaction_123"
    store.record_learning_event.return_value = {}
    return store


# =====================================================================
# 1. Resilient Zone query maps to the correct relevant objective
# =====================================================================

def test_resilient_zone_query_maps_to_relevant_objective(tool_registry, mock_learning_store):
    """Verifies that asking about the resilient zone selects obj_01_02 or obj_01_06 instead of obj_01_01."""
    planner = LearningPlannerAgent(tool_registry=tool_registry)

    planner_input = LearningPlannerInput(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        student_question="Explain the resilient zone using an A/L exam pressure example",
        current_skill_id="skill_01",
    )

    state = AgentState(student_id="STU-001")

    with patch("agents.tools.progress_tools.get_default_learning_store", return_value=mock_learning_store):
        output = planner.execute(planner_input, state=state)

    assert output.skill["id"] == "skill_01"
    # Must NOT select obj_01_01 (body sensations); must select resilient zone objective (obj_01_02 or obj_01_06)
    assert output.learning_objective["id"] in {"obj_01_02", "obj_01_06"}
    assert output.learning_objective["id"] != "obj_01_01"

    # Must contain relevant approved SEEK chunks
    assert len(output.relevant_chunks) > 0
    # Relevant chunks must be about the resilient zone / triggers
    chunk_text = " ".join(c.get("content", "") + " " + c.get("title", "") for c in output.relevant_chunks)
    assert "resilient" in chunk_text.lower() or "zone" in chunk_text.lower()


# =====================================================================
# 2. Retrieval does not discard all relevant chunks for unrelated pending objective
# =====================================================================

def test_retrieval_does_not_discard_chunks_for_unrelated_pending_objective(tool_registry):
    """When strict objective filtering yields zero chunks (e.g. obj_01_01 requested for resilient zone),

    retrieve_approved_seek_content retries within the approved skill and returns relevant chunks.
    """
    query = "Explain the resilient zone using an A/L exam pressure example"

    # Explicitly query with skill_01 but unrelated objective_id obj_01_01
    chunks = retrieve_approved_seek_content(
        query=query,
        skill_id="skill_01",
        objective_id="obj_01_01",
        limit=5,
    )

    # Chunks must not be empty!
    assert len(chunks) > 0
    assert len(chunks) <= 5

    # All returned chunks must belong to the approved skill_01
    for c in chunks:
        assert c["skill_id"] == "skill_01"
        assert "SEEK" in str(c["source"])
        assert c["id"] is not None

    # Content must be relevant to the query (resilient zone)
    combined = " ".join(c["content"] + " " + c["title"] for c in chunks).lower()
    assert "resilient zone" in combined or "resilient" in combined


# =====================================================================
# 3. Student progress is not modified by ordinary topic questions
# =====================================================================

def test_student_progress_not_modified_by_topic_questions(tool_registry, mock_learning_store):
    """Student asking questions must never complete objectives or mutate curriculum progress."""
    planner = LearningPlannerAgent(tool_registry=tool_registry)

    planner_input = LearningPlannerInput(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        student_question="Explain the resilient zone using an A/L exam pressure example",
        current_skill_id="skill_01",
    )

    with patch("agents.tools.progress_tools.get_default_learning_store", return_value=mock_learning_store):
        output = planner.execute(planner_input)

    # Progress mutation functions must NEVER be called by planner
    mock_learning_store.complete_item.assert_not_called()
    mock_learning_store.complete_objective.assert_not_called()
    assert output.pending_objectives_count > 0


# =====================================================================
# 4. Unmatched questions preserve existing progress-based behavior
# =====================================================================

def test_unmatched_questions_preserve_progress_based_behavior(tool_registry, mock_learning_store):
    """When a question does not match any approved topic, planner preserves existing progress-based behavior."""
    planner = LearningPlannerAgent(tool_registry=tool_registry)

    planner_input = LearningPlannerInput(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        student_question="How do I bake sourdough bread?",
        current_skill_id="skill_01",
    )

    with patch("agents.tools.progress_tools.get_default_learning_store", return_value=mock_learning_store):
        output = planner.execute(planner_input)

    # Must preserve current skill and next pending objective
    assert output.skill["id"] == "skill_01"
    assert output.learning_objective["id"] == "obj_01_01"


# =====================================================================
# 5. Unapproved content is never returned
# =====================================================================

def test_unapproved_content_is_never_returned():
    """All chunks must originate from the authoritative SEEK Knowledge Base; invalid inputs are rejected."""
    kb = get_knowledge_base()
    valid_ids = {node["id"] for node in kb if isinstance(node, dict) and "id" in node}

    chunks = retrieve_approved_seek_content(
        query="Explain the resilient zone using an A/L exam pressure example",
        skill_id="skill_01",
        limit=5,
    )

    for c in chunks:
        assert c["id"] in valid_ids
        assert "SEEK" in str(c["source"])
        assert c["skill_id"] == "skill_01"

    # Completely invalid skill or objective IDs must raise ToolValidationError
    with pytest.raises(ToolValidationError):
        retrieve_approved_seek_content(query="test", skill_id="unapproved_skill")

    with pytest.raises(ToolValidationError):
        retrieve_approved_seek_content(query="test", objective_id="unapproved_objective")


# =====================================================================
# 6. End-to-End Test Question: Resilient Zone with A/L exam pressure
# =====================================================================

def test_end_to_end_resilient_zone_exam_pressure(mock_learning_store):
    """Tests the full orchestration flow with the audited user test question."""
    orch = EmpathyOrchestrator(learning_store=mock_learning_store)

    response, code = orch.orchestrate_turn(
        student_id="STU-001",
        authenticated_student_id="STU-001",
        message="Explain the resilient zone using an A/L exam pressure example",
    )

    assert code == 200
    assert response.get("skill_id") == "skill_01"
    assert response.get("objective_id") in {"obj_01_02", "obj_01_06"}

    # Tutor response must NOT be empty or an insufficient-information fallback
    answer = response.get("answer") or response.get("educational_response") or ""
    assert len(answer.strip()) > 0
    assert "don't have enough approved information" not in answer.lower()

    # Must contain resilient zone concepts
    assert "resilient" in answer.lower() or "zone" in answer.lower()

    # Steps must adhere to 4-stage contract
    steps = response.get("steps", [])
    assert len(steps) == 4

    # Approved Resilient Zone activity must be selected
    rec_activity = response.get("learning_context", {}).get("recommended_activity")
    assert rec_activity is not None
    assert rec_activity.get("id") == "activity_01_02"
    assert rec_activity.get("title") == "What Would You Do?"

    act_step = steps[2]
    assert act_step.get("type") == "activity"
    assert act_step.get("activity") is not None
    assert act_step["activity"]["id"] == "activity_01_02"
    assert act_step["activity"]["title"] == "What Would You Do?"

    # No objective marked completed
    mock_learning_store.complete_objective.assert_not_called()
    mock_learning_store.complete_item.assert_not_called()


# =====================================================================
# 7. Approved Activity Selection & Supplementary Mapping
# =====================================================================

def test_approved_activity_selection_and_mapping(tool_registry):
    """Verifies that approved activities are correctly selected directly and via supplementary objectives."""
    from agents.tools.curriculum_tools import get_learning_activity

    # Direct canonical objective
    activity_direct = get_learning_activity(skill_id="skill_01", objective_id="obj_01_02")
    assert activity_direct["id"] == "activity_01_02"
    assert activity_direct["title"] == "What Would You Do?"
    assert activity_direct["skill_id"] == "skill_01"
    assert activity_direct.get("scenario") is not None
    assert "Your friend is sitting alone" in activity_direct["scenario"]
    assert len(activity_direct.get("options", [])) == 3
    assert activity_direct.get("best_answer") == "B"
    assert "correct" in activity_direct.get("feedback", {})

    # Supplementary objective activity (obj_01_06 has authentic reflection_01_06 or fallback activity_01_02)
    activity_mapped = get_learning_activity(skill_id="skill_01", objective_id="obj_01_06")
    assert activity_mapped["id"] in {"reflection_01_06", "activity_01_02"}
    assert activity_mapped.get("title") is not None


# =====================================================================
# 8. Missing Activity Graceful Handling
# =====================================================================

def test_missing_activity_handled_gracefully(tool_registry, mock_learning_store):
    """When an objective genuinely has no approved activity, ToolValidationError is raised and planner handles it gracefully."""
    from agents.tools.curriculum_tools import get_learning_activity

    empty_structure = {
        "chapters": [
            {
                "chapter_id": "skill_01",
                "objectives": [{"objective_id": "obj_01_01", "items": []}],
            }
        ]
    }
    with pytest.raises(ToolValidationError, match="No approved activities found"):
        get_learning_activity(
            skill_id="skill_01",
            objective_id="obj_01_01",
            progress_structure=empty_structure,
        )

    # When tool registry returns no activity, planner gracefully falls back
    planner = LearningPlannerAgent(tool_registry=tool_registry)
    with patch.object(planner, "call_tool") as mock_tool:
        mock_tool.side_effect = lambda tool_name, **kwargs: (
            [] if tool_name == "retrieve_approved_seek_content"
            else None if tool_name == "get_learning_activity"
            else [{"skill_id": "skill_01", "objective_id": "obj_01_01"}] if tool_name == "get_pending_objectives"
            else {"skill_id": "skill_01", "title": "Skill 1"} if tool_name == "get_current_skill"
            else {"overall_completed_percentage": 0}
        )
        planner_input = LearningPlannerInput(
            student_id="STU-001",
            authenticated_student_id="STU-001",
            student_question="General greeting without specific activity match",
            current_skill_id="skill_01",
        )
        output = planner.execute(planner_input)

    assert output.recommended_activity is None
    assert output.recommended_next_action == "introduce_concept_and_check"


# =====================================================================
# 9. No Unapproved or Invented Activities Returned
# =====================================================================

def test_no_unapproved_or_invented_activities():
    """Agents cannot retrieve fake, unapproved, or invented activity IDs."""
    from agents.tools.curriculum_tools import get_learning_activity

    with pytest.raises(ToolValidationError, match="Invalid skill_id"):
        get_learning_activity(skill_id="invented_skill", objective_id="obj_01_01")

    with pytest.raises(ToolValidationError, match="not an approved activity"):
        get_learning_activity(skill_id="skill_01", activity_id="nonexistent_activity_xyz")


# =====================================================================
# 10. Post-Quiz Continuity & SEEK Content Alignment for obj_01_03
# =====================================================================

def test_post_quiz_continuity_obj_01_03_seek_alignment(tool_registry):
    """Verifies that concepts for obj_01_03 (sympathetic/parasympathetic) are grounded in approved SEEK content."""
    from agents.tools.curriculum_tools import retrieve_approved_seek_content

    chunks = retrieve_approved_seek_content(
        query="sympathetic parasympathetic nervous system calming alerting",
        skill_id="skill_01",
        objective_id="obj_01_03",
    )
    assert len(chunks) > 0
    all_content = " ".join(c["content"].lower() for c in chunks)
    assert "parasympathetic" in all_content
    assert "sympathetic" in all_content

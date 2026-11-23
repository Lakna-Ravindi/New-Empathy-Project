"""Unit tests for Agent State Management layer."""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from agents.state import (
    AgentState,
    AgentStateManager,
    AgentStateError,
    InvalidStateTransitionError,
    UnauthorizedAccessError,
    UnauthorizedProgressMutationError,
    InvalidEmotionalContextError,
    StateValidationError,
    LearningStage,
    ALL_CANONICAL_OBJECTIVE_IDS,
    validate_conversational_emotional_context,
)


# =====================================================================
# State Creation & Validation Tests
# =====================================================================


def test_agent_state_creation_defaults():
    state = AgentState(student_id="STU-001")
    assert state.student_id == "STU-001"
    assert state.conversation_id == "default"
    assert state.current_message == ""
    assert state.conversation_history == []
    assert state.detected_intent is None
    assert state.current_skill_id is None
    assert state.current_objective_id is None
    assert state.completed_objectives == []
    assert state.pending_objectives == []
    assert state.current_learning_stage == LearningStage.INITIALIZED
    assert state.emotional_context["emotion"] == "calm"
    assert state.emotional_context["is_clinical_diagnosis"] is False
    assert state.selected_agent is None
    assert state.tool_results == []
    assert state.recommended_next_action is None
    assert state.safety_status["qualifies"] is False
    assert state.response is None
    assert state.error_status is None
    assert isinstance(state.execution_trace, list)


def test_agent_state_creation_explicit():
    state = AgentState(
        student_id="STU-ABC",
        conversation_id="conv-123",
        current_message="How do I practice tracking?",
        answer_style="long",
        current_skill_id="skill_01",
        current_objective_id="obj_01_04",
        completed_objectives=["obj_01_01", "obj_01_02"],
        pending_objectives=["obj_01_03", "obj_01_04"],
    )
    assert state.student_id == "STU-ABC"
    assert state.conversation_id == "conv-123"
    assert state.current_message == "How do I practice tracking?"
    assert state.answer_style == "long"
    assert state.current_skill_id == "skill_01"
    assert state.current_objective_id == "obj_01_04"
    assert state.completed_objectives == ["obj_01_01", "obj_01_02"]
    assert state.pending_objectives == ["obj_01_03", "obj_01_04"]


def test_agent_state_validation_invalid_ids():
    with pytest.raises(StateValidationError, match="student_id must be a non-empty string"):
        AgentState(student_id="")

    with pytest.raises(StateValidationError, match="student_id must be a non-empty string"):
        AgentState(student_id=None)  # type: ignore

    with pytest.raises(StateValidationError, match="conversation_id must be a non-empty string"):
        AgentState(student_id="STU-001", conversation_id="")


def test_agent_state_validation_invalid_learning_stage():
    with pytest.raises(StateValidationError, match="Invalid learning stage"):
        AgentState(student_id="STU-001", current_learning_stage="non_existent_stage")


# =====================================================================
# State Updates Tests
# =====================================================================


def test_agent_state_updates():
    state = AgentState(student_id="STU-001")

    # Update turn details
    state.current_message = "I am stressed about exams."
    state.detected_intent = "emotional_distress"
    state.selected_agent = "emotional_support_agent"

    # Add conversation history
    state.add_turn_to_history("student", "I am stressed about exams.", extra={"secret": "filter_me", "turn_id": 1})
    assert len(state.conversation_history) == 1
    assert state.conversation_history[0]["content"] == "I am stressed about exams."
    assert "secret" not in state.conversation_history[0]  # Sensitive info filtered
    assert state.conversation_history[0]["turn_id"] == 1

    # Record tool result
    state.record_tool_result("curriculum_retrieval", "tutor_agent", {"chunks": ["chunk1"]})
    assert len(state.tool_results) == 1
    assert state.tool_results[0]["tool_name"] == "curriculum_retrieval"
    assert state.tool_results[0]["result"] == {"chunks": ["chunk1"]}

    # Set response
    response_payload = {
        "answer": "It is completely normal to feel stressed.",
        "steps": [{"step": 1, "type": "direct_answer", "content": "..."}],
    }
    state.set_response(response_payload)
    assert state.response == response_payload

    # Set error
    state.set_error("API rate limit exceeded", stage="tutoring", code=429)
    assert state.error_status["error"] == "API rate limit exceeded"
    assert state.error_status["code"] == 429
    assert state.error_status["stage"] == "tutoring"


# =====================================================================
# Emotional Context Tests (No Medical Diagnosis)
# =====================================================================


def test_conversational_emotional_context_allowed():
    state = AgentState(student_id="STU-001")
    valid_context = {
        "emotion": "anxious",
        "valence": "negative",
        "intensity": "moderate",
        "topic_focus": "exam preparation",
        "readiness_for_learning": True,
    }
    state.update_emotional_context(valid_context)
    assert state.emotional_context["emotion"] == "anxious"
    assert state.emotional_context["topic_focus"] == "exam preparation"
    assert state.emotional_context["is_clinical_diagnosis"] is False


@pytest.mark.parametrize("prohibited_term", [
    "major depressive disorder",
    "bipolar disorder",
    "schizophrenia",
    "clinical depression",
    "PTSD",
    "dsm-5",
    "SSRI medication",
    "psychiatric diagnosis",
    "manic episode",
    "psychosis",
])
def test_conversational_emotional_context_rejects_medical_diagnoses(prohibited_term):
    state = AgentState(student_id="STU-001")

    with pytest.raises(InvalidEmotionalContextError, match="Forbidden clinical/medical diagnostic term"):
        state.update_emotional_context({
            "emotion": "sad",
            "notes": f"Student exhibits symptoms of {prohibited_term}",
        })

    with pytest.raises(InvalidEmotionalContextError):
        AgentState(
            student_id="STU-001",
            emotional_context={"diagnostic_label": prohibited_term},
        )


def test_emotional_context_cannot_be_marked_clinical():
    state = AgentState(student_id="STU-001")
    with pytest.raises(InvalidEmotionalContextError, match="cannot be marked as a clinical diagnosis"):
        state.update_emotional_context({"is_clinical_diagnosis": True})


# =====================================================================
# State Transition Validation Tests
# =====================================================================


def test_valid_state_transitions():
    state = AgentState(student_id="STU-001")
    assert state.current_learning_stage == LearningStage.INITIALIZED

    # Happy path flow
    state.transition_to(LearningStage.SAFETY_CHECK, reason="User query received")
    assert state.current_learning_stage == LearningStage.SAFETY_CHECK

    state.transition_to(LearningStage.INTENT_ROUTING, reason="Passed safety gate")
    assert state.current_learning_stage == LearningStage.INTENT_ROUTING

    state.transition_to(LearningStage.LEARNING_PLANNING, reason="Routed to SEEK curriculum")
    assert state.current_learning_stage == LearningStage.LEARNING_PLANNING

    state.transition_to(LearningStage.TUTORING, reason="Planner selected objective obj_01_01")
    assert state.current_learning_stage == LearningStage.TUTORING

    state.transition_to(LearningStage.RESPONSE_VALIDATION, reason="Generated 4-step response")
    assert state.current_learning_stage == LearningStage.RESPONSE_VALIDATION

    state.transition_to(LearningStage.COMPLETED, reason="Response schema validated")
    assert state.current_learning_stage == LearningStage.COMPLETED

    assert len(state.execution_trace) == 6
    assert state.execution_trace[0]["from_stage"] == "initialized"
    assert state.execution_trace[0]["to_stage"] == "safety_check"


def test_invalid_state_transitions():
    state = AgentState(student_id="STU-001")

    # Cannot jump directly from INITIALIZED to COMPLETED or TUTORING
    with pytest.raises(InvalidStateTransitionError, match="Invalid state transition"):
        state.transition_to(LearningStage.COMPLETED)

    with pytest.raises(InvalidStateTransitionError, match="Invalid state transition"):
        state.transition_to(LearningStage.TUTORING)

    # Valid step
    state.transition_to(LearningStage.SAFETY_CHECK)

    # Cannot jump from SAFETY_CHECK to TUTORING
    with pytest.raises(InvalidStateTransitionError):
        state.transition_to(LearningStage.TUTORING)


# =====================================================================
# Progress Mutation Security (RBAC) Tests
# =====================================================================


def test_progress_mutation_by_authorized_agent():
    state = AgentState(
        student_id="STU-001",
        current_skill_id="skill_01",
        pending_objectives=["obj_01_01", "obj_01_02"],
    )

    verification_result = {
        "verified": True,
        "score": 100,
        "feedback": "Correct reflection on somatic tracking.",
    }

    state.record_objective_completion(
        objective_id="obj_01_01",
        authorizing_agent="assessment_agent",
        verification_result=verification_result,
    )

    assert "obj_01_01" in state.completed_objectives
    assert "obj_01_01" not in state.pending_objectives
    assert any(t.get("action") == "objective_completed" for t in state.execution_trace)


def test_progress_mutation_by_unauthorized_agent_rejected():
    state = AgentState(student_id="STU-001", pending_objectives=["obj_01_01"])

    unauthorized_agents = ["emotional_support_agent", "tutor_agent", "planner_agent", "frontend", "client"]
    for agent in unauthorized_agents:
        with pytest.raises(UnauthorizedProgressMutationError, match="is not authorized to mutate curriculum progress"):
            state.record_objective_completion(
                objective_id="obj_01_01",
                authorizing_agent=agent,
                verification_result={"verified": True, "score": 100},
            )

    assert "obj_01_01" not in state.completed_objectives


def test_progress_mutation_unverified_rejected():
    state = AgentState(student_id="STU-001", pending_objectives=["obj_01_01"])

    with pytest.raises(UnauthorizedProgressMutationError, match="verification failed or score < 90"):
        state.record_objective_completion(
            objective_id="obj_01_01",
            authorizing_agent="assessment_agent",
            verification_result={"verified": False, "score": 60},
        )

    assert "obj_01_01" not in state.completed_objectives


def test_progress_mutation_non_canonical_objective_rejected():
    state = AgentState(student_id="STU-001")

    with pytest.raises(UnauthorizedProgressMutationError, match="is not a canonical SEEK objective"):
        state.record_objective_completion(
            objective_id="obj_nonexistent_99",
            authorizing_agent="assessment_agent",
            verification_result={"verified": True, "score": 100},
        )


# =====================================================================
# State Isolation & Authentication Tests
# =====================================================================


def test_state_isolation_between_students():
    manager = AgentStateManager()

    state_alice = manager.get_or_create_state("STU-ALICE", conversation_id="conv-1")
    state_bob = manager.get_or_create_state("STU-BOB", conversation_id="conv-1")

    state_alice.current_message = "Alice question"
    state_alice.completed_objectives.append("obj_01_01")

    state_bob.current_message = "Bob question"
    state_bob.completed_objectives.append("obj_02_01")

    # Verify completely isolated
    assert state_alice.student_id == "STU-ALICE"
    assert state_alice.current_message == "Alice question"
    assert state_alice.completed_objectives == ["obj_01_01"]

    assert state_bob.student_id == "STU-BOB"
    assert state_bob.current_message == "Bob question"
    assert state_bob.completed_objectives == ["obj_02_01"]

    # Retrieve again from manager
    retrieved_alice = manager.get_state("STU-ALICE", "conv-1")
    retrieved_bob = manager.get_state("STU-BOB", "conv-1")

    assert retrieved_alice is state_alice
    assert retrieved_bob is state_bob
    assert retrieved_alice.current_message != retrieved_bob.current_message


def test_state_isolation_between_conversations():
    manager = AgentStateManager()

    state_c1 = manager.get_or_create_state("STU-001", conversation_id="conv-1")
    state_c2 = manager.get_or_create_state("STU-001", conversation_id="conv-2")

    state_c1.current_message = "Topic 1"
    state_c2.current_message = "Topic 2"

    assert state_c1.current_message == "Topic 1"
    assert state_c2.current_message == "Topic 2"
    assert manager.list_conversations("STU-001") == ["conv-1", "conv-2"]


def test_reject_untrusted_frontend_student_id():
    manager = AgentStateManager()

    # Authenticated token identity is STU-GENUINE, but frontend attempts to spoof STU-VICTIM
    with pytest.raises(UnauthorizedAccessError, match="Security violation: Untrusted student_id"):
        manager.get_or_create_state(
            authenticated_student_id="STU-GENUINE",
            conversation_id="default",
            untrusted_student_id="STU-VICTIM",
        )

    # Matching untrusted ID succeeds
    state = manager.get_or_create_state(
        authenticated_student_id="STU-GENUINE",
        conversation_id="default",
        untrusted_student_id="STU-GENUINE",
    )
    assert state.student_id == "STU-GENUINE"


def test_unauthorized_state_save_prevented():
    manager = AgentStateManager()
    state_alice = AgentState(student_id="STU-ALICE")

    with pytest.raises(UnauthorizedAccessError, match="Unauthorized state modification"):
        manager.save_state(state_alice, authenticated_student_id="STU-EVE")


def test_state_manager_cleanup():
    manager = AgentStateManager()
    manager.get_or_create_state("STU-001", "conv-1")
    manager.get_or_create_state("STU-001", "conv-2")
    manager.get_or_create_state("STU-002", "conv-1")

    assert len(manager.list_conversations("STU-001")) == 2
    deleted = manager.delete_state("STU-001", "conv-1")
    assert deleted is True
    assert len(manager.list_conversations("STU-001")) == 1

    cleared = manager.clear_student_states("STU-001")
    assert cleared == 1
    assert len(manager.list_conversations("STU-001")) == 0
    # STU-002 untouched
    assert manager.get_state("STU-002", "conv-1") is not None


# =====================================================================
# Serialization & Persistence Integration Tests
# =====================================================================


def test_agent_state_serialization_roundtrip():
    original = AgentState(
        student_id="STU-001",
        conversation_id="conv-99",
        current_message="Hello",
        current_skill_id="skill_01",
        current_objective_id="obj_01_01",
        completed_objectives=["obj_01_01"],
        current_learning_stage=LearningStage.TUTORING,
    )
    serialized = original.to_dict()
    assert isinstance(serialized, dict)
    assert serialized["student_id"] == "STU-001"
    assert serialized["current_learning_stage"] == "tutoring"

    restored = AgentState.from_dict(serialized)
    assert restored.student_id == original.student_id
    assert restored.conversation_id == original.conversation_id
    assert restored.current_learning_stage == LearningStage.TUTORING
    assert restored.completed_objectives == ["obj_01_01"]


def test_hydrate_from_learning_store():
    manager = AgentStateManager()
    state = manager.get_or_create_state("STU-001")

    mock_learning_store = MagicMock()
    mock_learning_store.list_interactions.return_value = [
        {"question": "What is empathy?", "response": "Empathy is feeling with others."},
    ]
    mock_learning_store.get_all_progress.return_value = [
        {"skill_id": "skill_01", "completed_objective_ids": ["obj_01_01", "obj_01_02"]},
    ]

    manager.hydrate_from_learning_store(state, mock_learning_store)

    assert len(state.conversation_history) == 2
    assert state.conversation_history[0]["content"] == "What is empathy?"
    assert state.conversation_history[1]["content"] == "Empathy is feeling with others."
    assert "obj_01_01" in state.completed_objectives
    assert "obj_01_02" in state.completed_objectives
    assert "obj_01_01" not in state.pending_objectives


def test_persist_turn_to_existing_collections():
    manager = AgentStateManager()
    state = manager.get_or_create_state(
        "STU-001",
        current_skill_id="skill_01",
        current_objective_id="obj_01_04",
    )
    state.current_message = "How do I practice grounding?"
    state.metadata["recommended_activity"] = {"progress_item_id": "act_01_04"}
    state.set_response({
        "educational_response": "Try contact grounding practice.",
        "steps": [{"step": 1, "type": "direct_answer", "content": "..."}],
    })

    mock_learning_store = MagicMock()
    mock_learning_store.save_interaction.return_value = "interaction-xyz-123"

    interaction_id = manager.persist_turn(state, mock_learning_store)

    assert interaction_id == "interaction-xyz-123"
    mock_learning_store.save_interaction.assert_called_once()
    mock_learning_store.record_learning_event.assert_called_once_with(
        student_id="STU-001",
        skill_id="skill_01",
        item_id="act_01_04",
        objective_id="obj_01_04",
        source="chatbot",
        status="introduced",
    )

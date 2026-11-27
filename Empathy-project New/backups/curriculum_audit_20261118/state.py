"""Agent State Management Layer for Empathy Guru.

This module provides the structured AgentState and AgentStateManager adhering to
the approved UNESCO MGIEP SEEK Hybrid Multi-Agent architecture.

Key features:
1. Strict session and conversation isolation per authenticated student.
2. Sourcing student identity exclusively from JWT/auth, rejecting untrusted frontend IDs.
3. Conversational emotional context validation with strict disallowance of medical diagnoses.
4. Deterministic state transition validation across the learning lifecycle.
5. RBAC-guarded curriculum progress mutation preventing unauthorized progress alterations.
6. Seamless integration with existing LearningStore and MongoDB collections without creating new collections.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

logger = logging.getLogger(__name__)


# =====================================================================
# Exceptions
# =====================================================================


class AgentStateError(Exception):
    """Base exception for all agent state operations."""
    pass


class InvalidStateTransitionError(AgentStateError):
    """Raised when an invalid state transition is attempted."""
    pass


class UnauthorizedAccessError(AgentStateError):
    """Raised when access to student or conversation state is unauthorized."""
    pass


class UnauthorizedProgressMutationError(AgentStateError):
    """Raised when an unauthorized agent or unverified action attempts to mutate progress."""
    pass


class InvalidEmotionalContextError(AgentStateError):
    """Raised when emotional context contains prohibited medical or diagnostic data."""
    pass


class StateValidationError(AgentStateError):
    """Raised when state data fails structure or schema validation."""
    pass


# =====================================================================
# Enums & Canonical Constants
# =====================================================================


class LearningStage(str, Enum):
    """Lifecycle stages for multi-agent learning orchestration."""
    INITIALIZED = "initialized"
    SAFETY_CHECK = "safety_check"
    INTENT_ROUTING = "intent_routing"
    EMOTIONAL_SUPPORT = "emotional_support"
    LEARNING_PLANNING = "learning_planning"
    TUTORING = "tutoring"
    ASSESSMENT = "assessment"
    RESPONSE_VALIDATION = "response_validation"
    COMPLETED = "completed"
    RESTRICTED = "restricted"
    ERROR = "error"


# Allowed state transitions enforcing workflow integrity
ALLOWED_STAGE_TRANSITIONS: Dict[LearningStage, Set[LearningStage]] = {
    LearningStage.INITIALIZED: {
        LearningStage.SAFETY_CHECK,
        LearningStage.INTENT_ROUTING,
        LearningStage.ERROR,
    },
    LearningStage.SAFETY_CHECK: {
        LearningStage.INTENT_ROUTING,
        LearningStage.RESTRICTED,
        LearningStage.ERROR,
    },
    LearningStage.INTENT_ROUTING: {
        LearningStage.EMOTIONAL_SUPPORT,
        LearningStage.LEARNING_PLANNING,
        LearningStage.TUTORING,
        LearningStage.ASSESSMENT,
        LearningStage.COMPLETED,
        LearningStage.ERROR,
    },
    LearningStage.EMOTIONAL_SUPPORT: {
        LearningStage.LEARNING_PLANNING,
        LearningStage.TUTORING,
        LearningStage.RESPONSE_VALIDATION,
        LearningStage.COMPLETED,
        LearningStage.ERROR,
    },
    LearningStage.LEARNING_PLANNING: {
        LearningStage.TUTORING,
        LearningStage.ASSESSMENT,
        LearningStage.RESPONSE_VALIDATION,
        LearningStage.COMPLETED,
        LearningStage.ERROR,
    },
    LearningStage.TUTORING: {
        LearningStage.RESPONSE_VALIDATION,
        LearningStage.ASSESSMENT,
        LearningStage.COMPLETED,
        LearningStage.ERROR,
    },
    LearningStage.ASSESSMENT: {
        LearningStage.RESPONSE_VALIDATION,
        LearningStage.COMPLETED,
        LearningStage.ERROR,
    },
    LearningStage.RESPONSE_VALIDATION: {
        LearningStage.COMPLETED,
        LearningStage.TUTORING,  # Retry / fallback loop
        LearningStage.ERROR,
    },
    LearningStage.COMPLETED: {
        LearningStage.INITIALIZED,  # Reset for next turn
        LearningStage.SAFETY_CHECK,
    },
    LearningStage.RESTRICTED: {
        LearningStage.INITIALIZED,  # Upon clinical review unlock
        LearningStage.ERROR,
    },
    LearningStage.ERROR: {
        LearningStage.INITIALIZED,  # Reset for retry
        LearningStage.COMPLETED,    # Fallback response completed
    },
}

# Agents authorized to commit curriculum progress
AUTHORIZED_PROGRESS_MUTATORS: Set[str] = {
    "assessment_agent",
    "assessment",
    "system_admin",
}

# Prohibited medical diagnostic, psychopathological, or clinical terms.
# Conversational emotional context must never act as a psychiatric diagnosis.
PROHIBITED_MEDICAL_TERMS: Set[str] = {
    "depression",
    "depressive",
    "major depressive disorder",
    "mdd",
    "bipolar",
    "mania",
    "manic",
    "schizophrenia",
    "schizoaffective",
    "psychosis",
    "psychotic",
    "ptsd",
    "post-traumatic stress",
    "post traumatic stress",
    "ocd",
    "obsessive-compulsive",
    "obsessive compulsive",
    "borderline",
    "bpd",
    "anorexia",
    "bulimia",
    "dsm",
    "dsm-5",
    "dsm-iv",
    "icd-10",
    "icd-11",
    "clinical diagnosis",
    "medical diagnosis",
    "psychiatric diagnosis",
    "antidepressant",
    "antipsychotic",
    "ssri",
    "snri",
    "benzodiazepine",
    "pharmacotherapy",
    "psychopathology",
}

# Canonical SEEK Objectives across all 8 skills
ALL_CANONICAL_OBJECTIVE_IDS: Set[str] = {
    f"obj_{skill:02d}_{obj:02d}"
    for skill in range(1, 9)
    for obj in range(1, 5)
}
# Add supplementary objectives present in curriculum definition
ALL_CANONICAL_OBJECTIVE_IDS.update({
    "obj_01_05", "obj_01_06",
    "obj_02_05",
    "obj_03_05", "obj_03_06",
    "obj_04_05", "obj_04_06", "obj_04_07",
    "obj_05_05",
    "obj_06_05", "obj_06_06",
    "obj_07_05", "obj_07_06",
    "obj_08_05",
})

ALL_CANONICAL_SKILL_IDS: Set[str] = {f"skill_{i:02d}" for i in range(1, 9)}


# =====================================================================
# Emotional Context Validation
# =====================================================================


def validate_conversational_emotional_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """Validates that emotional context is strictly conversational and contains no medical diagnoses."""
    if not isinstance(context, dict):
        raise InvalidEmotionalContextError("emotional_context must be a dictionary.")

    # Recursively check string values and keys for prohibited medical diagnostic language
    def _check_string(val: str, field_name: str = ""):
        lowered = val.lower()
        for term in PROHIBITED_MEDICAL_TERMS:
            if term in lowered:
                raise InvalidEmotionalContextError(
                    f"Forbidden clinical/medical diagnostic term '{term}' detected in emotional_context "
                    f"field '{field_name}'. Emotional context must only capture conversational tone, "
                    f"not a medical or psychiatric diagnosis."
                )

    def _walk(obj: Any, parent: str = ""):
        if isinstance(obj, str):
            _check_string(obj, parent)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _check_string(str(k), f"{parent}.key:{k}" if parent else f"key:{k}")
                _walk(v, f"{parent}.{k}" if parent else str(k))
        elif isinstance(obj, (list, tuple, set)):
            for idx, item in enumerate(obj):
                _walk(item, f"{parent}[{idx}]")

    _walk(context)

    # Sanitize and ensure explicit non-clinical marker
    sanitized = copy.deepcopy(context)
    sanitized.setdefault("is_clinical_diagnosis", False)
    if sanitized.get("is_clinical_diagnosis") is True:
        raise InvalidEmotionalContextError("emotional_context cannot be marked as a clinical diagnosis.")

    return sanitized


# =====================================================================
# AgentState Dataclass
# =====================================================================


@dataclass
class AgentState:
    """Structured in-memory state for Empathy Guru multi-agent conversational turns.

    Maintains total isolation per (student_id, conversation_id), preserves existing
    persistence contracts, and prevents unauthorized curriculum mutations.
    """

    # Identifiers
    student_id: str
    conversation_id: str = "default"
    request_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Turn Inputs & History
    current_message: str = ""
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    answer_style: str = "short"  # "short" | "long"

    # Routing & Intent
    detected_intent: Optional[str] = None
    selected_agent: Optional[str] = None

    # Curriculum & Progress Context
    current_skill_id: Optional[str] = None
    current_objective_id: Optional[str] = None
    completed_objectives: List[str] = field(default_factory=list)
    pending_objectives: List[str] = field(default_factory=list)
    current_learning_stage: LearningStage = LearningStage.INITIALIZED

    # Context & Results
    emotional_context: Dict[str, Any] = field(default_factory=lambda: {
        "emotion": "calm",
        "valence": "neutral",
        "intensity": "moderate",
        "readiness_for_learning": True,
        "is_clinical_diagnosis": False,
    })
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    recommended_next_action: Optional[Dict[str, Any]] = None

    # Safety & Quality
    safety_status: Dict[str, Any] = field(default_factory=lambda: {
        "qualifies": False,
        "risk_level": "none",
        "restriction_active": False,
    })

    # Response & Diagnostics
    response: Optional[Dict[str, Any]] = None
    error_status: Optional[Dict[str, Any]] = None
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate state fields on construction."""
        if not self.student_id or not isinstance(self.student_id, str):
            raise StateValidationError("student_id must be a non-empty string.")
        if not self.conversation_id or not isinstance(self.conversation_id, str):
            raise StateValidationError("conversation_id must be a non-empty string.")

        # Ensure learning stage enum
        if isinstance(self.current_learning_stage, str):
            try:
                self.current_learning_stage = LearningStage(self.current_learning_stage)
            except ValueError as err:
                raise StateValidationError(f"Invalid learning stage: {self.current_learning_stage}") from err

        # Validate emotional context
        self.emotional_context = validate_conversational_emotional_context(self.emotional_context)

        # Validate skill and objective IDs if set
        if self.current_skill_id and self.current_skill_id not in ALL_CANONICAL_SKILL_IDS:
            logger.warning("Unrecognized skill_id: %s", self.current_skill_id)

    # -----------------------------------------------------------------
    # State Transition Management
    # -----------------------------------------------------------------

    def transition_to(self, target_stage: LearningStage | str, reason: Optional[str] = None) -> None:
        """Transitions the agent state to target_stage if permitted by workflow rules."""
        if isinstance(target_stage, str):
            try:
                target_stage = LearningStage(target_stage)
            except ValueError as err:
                raise InvalidStateTransitionError(f"Unknown learning stage '{target_stage}'.") from err

        current = self.current_learning_stage
        allowed = ALLOWED_STAGE_TRANSITIONS.get(current, set())

        if target_stage not in allowed:
            raise InvalidStateTransitionError(
                f"Invalid state transition: Cannot transition from '{current.value}' to '{target_stage.value}'. "
                f"Allowed target stages are: {[s.value for s in allowed]}"
            )

        previous_stage = self.current_learning_stage
        self.current_learning_stage = target_stage

        trace_entry = {
            "from_stage": previous_stage.value,
            "to_stage": target_stage.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason or "",
        }
        self.execution_trace.append(trace_entry)

    # -----------------------------------------------------------------
    # Emotional Context Management
    # -----------------------------------------------------------------

    def update_emotional_context(self, context: Dict[str, Any]) -> None:
        """Safely updates emotional context ensuring no clinical/medical diagnoses are recorded."""
        validated = validate_conversational_emotional_context(context)
        self.emotional_context.update(validated)

    # -----------------------------------------------------------------
    # Progress Mutation Security (RBAC)
    # -----------------------------------------------------------------

    def record_objective_completion(
        self,
        objective_id: str,
        authorizing_agent: str,
        verification_result: Dict[str, Any],
    ) -> None:
        """Records completion of a learning objective with strict RBAC and verification checks."""
        # 1. RBAC Check: Only authorized agents can commit progress
        normalized_agent = (authorizing_agent or "").strip().lower()
        if normalized_agent not in AUTHORIZED_PROGRESS_MUTATORS:
            raise UnauthorizedProgressMutationError(
                f"Agent '{authorizing_agent}' is not authorized to mutate curriculum progress. "
                f"Authorized roles are: {sorted(AUTHORIZED_PROGRESS_MUTATORS)}"
            )

        # 2. Canonical Objective Verification
        if objective_id not in ALL_CANONICAL_OBJECTIVE_IDS:
            raise UnauthorizedProgressMutationError(
                f"Objective ID '{objective_id}' is not a canonical SEEK objective."
            )

        # 3. Verification Result Proof Check
        if not isinstance(verification_result, dict):
            raise UnauthorizedProgressMutationError(
                "Progress mutation requires a valid verification_result dictionary."
            )

        is_verified = (
            verification_result.get("verified") is True
            or verification_result.get("success") is True
            or (
                isinstance(verification_result.get("score"), (int, float))
                and verification_result["score"] >= 90
            )
        )
        if not is_verified:
            raise UnauthorizedProgressMutationError(
                f"Objective '{objective_id}' cannot be marked complete: verification failed or score < 90."
            )

        # 4. Safe State Mutation
        if objective_id not in self.completed_objectives:
            self.completed_objectives.append(objective_id)

        if objective_id in self.pending_objectives:
            self.pending_objectives.remove(objective_id)

        self.execution_trace.append({
            "action": "objective_completed",
            "objective_id": objective_id,
            "authorizing_agent": authorizing_agent,
            "verification_summary": verification_result.get("feedback") or "Verified",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    # -----------------------------------------------------------------
    # History & Tool Management
    # -----------------------------------------------------------------

    def add_turn_to_history(self, role: str, content: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Appends a turn to conversation_history without storing sensitive info."""
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            # Filter out sensitive fields
            sanitized_extra = {
                k: v for k, v in extra.items()
                if k not in {"password", "token", "jwt", "authorization", "secret"}
            }
            entry.update(sanitized_extra)
        self.conversation_history.append(entry)

    def record_tool_result(
        self,
        tool_name: str,
        agent_role: str,
        result: Any,
        error: Optional[str] = None,
    ) -> None:
        """Records the output or error of an executed tool."""
        self.tool_results.append({
            "tool_name": tool_name,
            "agent_role": agent_role,
            "result": result,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def set_response(self, response_payload: Dict[str, Any]) -> None:
        """Sets the validated educational response."""
        if not isinstance(response_payload, dict):
            raise StateValidationError("response must be a dictionary.")
        self.response = response_payload

    def set_error(
        self,
        error_message: str,
        stage: Optional[str] = None,
        code: Optional[int] = None,
        details: Optional[Any] = None,
    ) -> None:
        """Records error status and updates trace."""
        self.error_status = {
            "error": error_message,
            "stage": stage or self.current_learning_stage.value,
            "code": code or 500,
            "details": details,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serializes AgentState into a JSON-compatible dictionary."""
        data = asdict(self)
        data["current_learning_stage"] = self.current_learning_stage.value
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentState:
        """Deserializes AgentState from a dictionary."""
        d = copy.deepcopy(data)
        if "timestamp" in d and isinstance(d["timestamp"], str):
            try:
                d["timestamp"] = datetime.fromisoformat(d["timestamp"])
            except ValueError:
                d["timestamp"] = datetime.now(timezone.utc)
        if "current_learning_stage" in d and isinstance(d["current_learning_stage"], str):
            d["current_learning_stage"] = LearningStage(d["current_learning_stage"])
        return cls(**d)


# =====================================================================
# AgentStateManager
# =====================================================================


class AgentStateManager:
    """Manages active AgentState instances with strict student & conversation isolation.

    Enforces:
    - Never trust student_id supplied by frontend; strictly use authenticated identity.
    - Zero state bleed between different students or distinct conversations.
    - Interoperability with LearningStore for hydrating and persisting interactions.
    - In-memory lifecycle without creating an unnecessary new database collection.
    """

    def __init__(self):
        self._states: Dict[str, AgentState] = {}
        self._lock = RLock()

    @staticmethod
    def _make_key(student_id: str, conversation_id: str) -> str:
        """Generates a composite, collision-free key ensuring strict student & conversation isolation."""
        return f"{student_id.strip()}::{conversation_id.strip()}"

    def get_or_create_state(
        self,
        authenticated_student_id: str,
        conversation_id: str = "default",
        untrusted_student_id: Optional[str] = None,
        **initial_kwargs,
    ) -> AgentState:
        """Retrieves or creates state enforcing the authenticated student identity.

        If the frontend supplies an untrusted_student_id that conflicts with the
        authenticated token identity, an UnauthorizedAccessError is raised.
        """
        if not authenticated_student_id or not isinstance(authenticated_student_id, str):
            raise UnauthorizedAccessError("A valid authenticated student_id is required.")

        auth_id = authenticated_student_id.strip()

        # Security check: frontend tampering rejection
        if untrusted_student_id is not None:
            untrusted = untrusted_student_id.strip()
            if untrusted and untrusted != auth_id:
                raise UnauthorizedAccessError(
                    f"Security violation: Untrusted student_id '{untrusted}' does not match "
                    f"authenticated JWT identity '{auth_id}'."
                )

        conv_id = (conversation_id or "default").strip()
        key = self._make_key(auth_id, conv_id)

        with self._lock:
            if key in self._states:
                return self._states[key]

            # Construct fresh isolated state
            new_state = AgentState(
                student_id=auth_id,
                conversation_id=conv_id,
                **initial_kwargs,
            )
            self._states[key] = new_state
            return new_state

    def get_state(
        self,
        authenticated_student_id: str,
        conversation_id: str = "default",
    ) -> Optional[AgentState]:
        """Retrieves existing state for the authenticated student only."""
        if not authenticated_student_id:
            raise UnauthorizedAccessError("authenticated_student_id is required.")
        key = self._make_key(authenticated_student_id, conversation_id)
        with self._lock:
            return self._states.get(key)

    def save_state(self, state: AgentState, authenticated_student_id: str) -> None:
        """Saves active state while asserting ownership by the authenticated student."""
        if state.student_id != authenticated_student_id:
            raise UnauthorizedAccessError(
                f"Unauthorized state modification: State belongs to student '{state.student_id}', "
                f"not authenticated caller '{authenticated_student_id}'."
            )
        key = self._make_key(state.student_id, state.conversation_id)
        with self._lock:
            self._states[key] = state

    def delete_state(self, authenticated_student_id: str, conversation_id: str = "default") -> bool:
        """Removes state for a specific student conversation."""
        key = self._make_key(authenticated_student_id, conversation_id)
        with self._lock:
            if key in self._states:
                del self._states[key]
                return True
            return False

    def clear_student_states(self, authenticated_student_id: str) -> int:
        """Removes all conversation states for the given student."""
        prefix = f"{authenticated_student_id.strip()}::"
        with self._lock:
            keys_to_delete = [k for k in self._states if k.startswith(prefix)]
            for k in keys_to_delete:
                del self._states[k]
            return len(keys_to_delete)

    def list_conversations(self, authenticated_student_id: str) -> List[str]:
        """Lists conversation IDs belonging to the authenticated student."""
        prefix = f"{authenticated_student_id.strip()}::"
        with self._lock:
            return [
                k[len(prefix):]
                for k in self._states
                if k.startswith(prefix)
            ]

    # -----------------------------------------------------------------
    # Persistence Interoperability with LearningStore
    # -----------------------------------------------------------------

    def hydrate_from_learning_store(
        self,
        state: AgentState,
        learning_store: Any,
        interaction_limit: int = 10,
    ) -> AgentState:
        """Populates state context from existing MongoDB collections via LearningStore.

        Does not create any new collections; reuses `learning_interactions` and `student_progress`.
        """
        student_id = state.student_id

        # 1. Hydrate dialogue history from recent interactions
        try:
            interactions = learning_store.list_interactions(student_id, limit=interaction_limit)
            if interactions:
                # Store interactions oldest to newest for conversational continuity
                for item in reversed(interactions):
                    question = item.get("question")
                    response = item.get("response")
                    if question:
                        state.add_turn_to_history("student", question)
                    if response:
                        state.add_turn_to_history("assistant", str(response))
        except Exception as err:
            logger.warning("Could not hydrate interaction history for %s: %s", student_id, err)

        # 2. Hydrate progress from existing student_progress collection
        try:
            all_progress = learning_store.get_all_progress(student_id) or []
            completed_set = set(state.completed_objectives)
            for prog in all_progress:
                completed_set.update(prog.get("completed_objective_ids", []))
            state.completed_objectives = sorted(completed_set)

            # Calculate pending objectives from canonical list
            state.pending_objectives = sorted(
                obj for obj in ALL_CANONICAL_OBJECTIVE_IDS
                if obj not in completed_set
            )
        except Exception as err:
            logger.warning("Could not hydrate progress for %s: %s", student_id, err)

        return state

    def persist_turn(
        self,
        state: AgentState,
        learning_store: Any,
    ) -> Optional[str]:
        """Persists the turn to the existing `learning_interactions` collection.

        Preserves 100% compatibility with the current frontend schema and server logic.
        """
        if not state.response:
            return None

        # Build learning_context compatible with existing schema
        learning_context = {
            "student_question": state.current_message,
            "detected_emotion": state.emotional_context.get("emotion", "calm"),
            "detected_terms": state.metadata.get("detected_terms", []),
            "status": state.detected_intent or "learning_path_selected",
            "skill": {
                "id": state.current_skill_id,
                "title": state.metadata.get("skill_title", ""),
            } if state.current_skill_id else None,
            "learning_objective": {
                "id": state.current_objective_id,
                "title": state.metadata.get("objective_title", ""),
            } if state.current_objective_id else None,
            "recommended_activity": state.metadata.get("recommended_activity"),
        }

        educational_response = state.response.get(
            "educational_response",
            state.response.get("answer", ""),
        )
        steps = state.response.get("steps")

        interaction_id = learning_store.save_interaction(
            student_id=state.student_id,
            learning_context=learning_context,
            educational_response=educational_response,
            learning_steps=steps,
        )

        # Record introduced/completed learning event if activity is present
        rec_activity = state.metadata.get("recommended_activity") or {}
        progress_item_id = rec_activity.get("progress_item_id")
        if progress_item_id and state.current_skill_id:
            try:
                learning_store.record_learning_event(
                    student_id=state.student_id,
                    skill_id=state.current_skill_id,
                    item_id=progress_item_id,
                    objective_id=state.current_objective_id,
                    source="chatbot",
                    status="introduced",
                )
            except Exception as err:
                logger.warning("Could not record learning event: %s", err)

        return str(interaction_id)

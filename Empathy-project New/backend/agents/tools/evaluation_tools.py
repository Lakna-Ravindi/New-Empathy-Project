"""Evaluation and interaction persistence tools.

Guarantees:
- Reuses LearningStore.get_interaction and LearningStore.save_interaction.
- Enforces authenticated student identity checks.
- evaluate_student_response is read-only (verification without mutation).
- save_learning_interaction safely commits interactions and learning events.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agents.state import (
    ALL_CANONICAL_OBJECTIVE_IDS,
    UnauthorizedAccessError,
)
from agents.tools.curriculum_tools import get_progress_structure
from agents.tools.progress_tools import (
    _assert_identity,
    get_default_learning_store,
    get_default_progress_service,
)
from agents.tools.registry import ToolValidationError
from learning.interaction_store import LearningStore
from learning.progress_service import ProgressService

logger = logging.getLogger(__name__)


def evaluate_student_response(
    student_id: str,
    authenticated_student_id: str,
    objective_id: str,
    submission_type: str,
    student_answer: Any,
    skill_id: Optional[str] = None,
    interaction_id: Optional[str] = None,
    item_id: Optional[str] = None,
    learning_store: Optional[LearningStore] = None,
    progress_service: Optional[ProgressService] = None,
) -> Dict[str, Any]:
    """Tool 6: Evaluates student answers, reflections, or practice checks.

    Reuses LearningStore.get_interaction and server-side practice check verification.
    Read-only: does not modify student progress directly.
    """
    _assert_identity(student_id, authenticated_student_id)

    if not objective_id or not isinstance(objective_id, str):
        raise ToolValidationError("objective_id must be a non-empty string.")

    norm_obj = objective_id.strip()
    if norm_obj not in ALL_CANONICAL_OBJECTIVE_IDS:
        raise ToolValidationError(f"Invalid objective_id '{objective_id}'. Must be a canonical SEEK objective.")

    norm_sub_type = (submission_type or "").strip().lower()
    if norm_sub_type not in {"quick_check", "reflection", "quiz"}:
        raise ToolValidationError(
            f"submission_type '{submission_type}' is invalid. Must be 'quick_check', 'reflection', or 'quiz'."
        )

    store = learning_store or get_default_learning_store()

    # 1. Quick Check Evaluation
    if norm_sub_type == "quick_check":
        if not interaction_id:
            raise ToolValidationError("interaction_id is required to verify a quick_check answer.")

        interaction = store.get_interaction(student_id, interaction_id)
        if not interaction:
            return {
                "verified": False,
                "correct": False,
                "feedback": "Original learning interaction was not found.",
                "score": 0.0,
                "objective_id": norm_obj,
                "submission_type": "quick_check",
            }

        learning_context = interaction.get("learning_context") or {}
        selected_skill = interaction.get("selected_skill") or learning_context.get("skill") or {}
        interaction_objective = (
            interaction.get("learning_objective")
            or learning_context.get("learning_objective")
            or {}
        )
        interaction_skill_id = (
            selected_skill.get("id")
            or selected_skill.get("skill_id")
            or selected_skill.get("chapter_id")
        )
        interaction_objective_id = (
            interaction_objective.get("id")
            or interaction_objective.get("objective_id")
        )

        # Validate skill_id against interaction context when available
        if skill_id and interaction_skill_id and str(interaction_skill_id).strip() != skill_id.strip():
            return {
                "verified": False,
                "correct": False,
                "feedback": f"Interaction skill '{interaction_skill_id}' does not match requested skill '{skill_id}'.",
                "score": 0.0,
                "objective_id": norm_obj,
                "interaction_id": interaction_id,
                "submission_type": "quick_check",
            }

        # Validate objective_id against interaction context when available
        if interaction_objective_id and str(interaction_objective_id).strip() != norm_obj:
            return {
                "verified": False,
                "correct": False,
                "feedback": f"Interaction objective '{interaction_objective_id}' does not match requested objective '{norm_obj}'.",
                "score": 0.0,
                "objective_id": norm_obj,
                "interaction_id": interaction_id,
                "submission_type": "quick_check",
            }

        learning_steps = interaction.get("learning_steps") or []
        practice_check = next(
            (s for s in learning_steps if isinstance(s, dict) and s.get("type") == "practice_check"),
            None,
        )
        if not practice_check:
            return {
                "verified": False,
                "correct": False,
                "feedback": "No practice check found in the referenced interaction.",
                "score": 0.0,
                "objective_id": norm_obj,
                "interaction_id": interaction_id,
                "submission_type": "quick_check",
            }

        correct_answer = practice_check.get("correct_answer")

        # Strict type-safe comparison matching server.py:1615–1620
        is_match = False
        if isinstance(correct_answer, bool) and isinstance(student_answer, bool):
            is_match = student_answer is correct_answer
        elif isinstance(correct_answer, str) and isinstance(student_answer, str):
            is_match = student_answer.strip().casefold() == correct_answer.strip().casefold()
        else:
            is_match = False

        return {
            "verified": is_match,
            "correct": is_match,
            "feedback": (
                "Great job! Your answer is correct."
                if is_match
                else "That answer is not quite right. Review the guidance and try again."
            ),
            "score": 100.0 if is_match else 0.0,
            "objective_id": norm_obj,
            "interaction_id": interaction_id,
            "submission_type": "quick_check",
        }

    # 2. Reflection Evaluation
    elif norm_sub_type == "reflection":
        text = str(student_answer or "").strip()
        word_count = len(text.split())

        if word_count < 3:
            return {
                "verified": False,
                "correct": False,
                "feedback": "Please share a more detailed reflection describing your thoughts or feelings.",
                "score": 30.0,
                "objective_id": norm_obj,
                "submission_type": "reflection",
                "can_complete_objective": False,
            }

        # Substantive reflection passed qualitative activity check,
        # but reflections alone can never verify objective completion.
        return {
            "verified": True,
            "correct": True,
            "feedback": "Thank you for reflecting thoughtfully on this concept.",
            "score": 100.0,
            "objective_id": norm_obj,
            "submission_type": "reflection",
            "can_complete_objective": False,
        }

    # 3. Quiz evaluation
    else:
        score = (
            float(student_answer)
            if isinstance(student_answer, (int, float)) and not isinstance(student_answer, bool)
            else 0.0
        )
        passed = score >= ProgressService.DEFAULT_QUIZ_PASSING_SCORE
        return {
            "verified": passed,
            "correct": passed,
            "feedback": f"Quiz scored {score}%. Passing threshold is {ProgressService.DEFAULT_QUIZ_PASSING_SCORE}%.",
            "score": score,
            "objective_id": norm_obj,
            "submission_type": "quiz",
        }


def save_learning_interaction(
    student_id: str,
    authenticated_student_id: str,
    question: str,
    response_payload: Dict[str, Any],
    learning_context: Dict[str, Any],
    learning_steps: Optional[List[Dict[str, Any]]] = None,
    learning_store: Optional[LearningStore] = None,
) -> str:
    """Tool 8: Persists an educational learning interaction and associated events.

    Reuses LearningStore.save_interaction and LearningStore.record_learning_event.
    Strictly asserts authenticated student identity.
    """
    _assert_identity(student_id, authenticated_student_id)

    if not question or not isinstance(question, str):
        raise ToolValidationError("question must be a non-empty string.")
    if not isinstance(response_payload, dict):
        raise ToolValidationError("response_payload must be a dictionary.")
    if not isinstance(learning_context, dict):
        raise ToolValidationError("learning_context must be a dictionary.")

    store = learning_store or get_default_learning_store()

    educational_response = response_payload.get(
        "educational_response",
        response_payload.get("answer", ""),
    )
    steps = learning_steps if learning_steps is not None else response_payload.get("steps")

    interaction_id = store.save_interaction(
        student_id=student_id,
        learning_context=learning_context,
        educational_response=educational_response,
        learning_steps=steps,
    )

    # If activity is recommended, record learning event
    rec_activity = learning_context.get("recommended_activity") or {}
    progress_item_id = rec_activity.get("progress_item_id")
    skill_info = learning_context.get("skill") or {}
    skill_id = skill_info.get("id") or skill_info.get("skill_id")

    if progress_item_id and skill_id:
        try:
            store.record_learning_event(
                student_id=student_id,
                skill_id=skill_id,
                item_id=progress_item_id,
                objective_id=rec_activity.get("progress_objective_id"),
                source="chatbot",
                status="introduced",
            )
        except Exception as err:
            logger.warning("Could not record learning event during interaction save: %s", err)

    return str(interaction_id)

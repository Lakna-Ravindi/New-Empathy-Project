"""Progress tools for inspecting, validating, and mutating student curriculum progress.

Guarantees:
- Reuses existing ProgressService and LearningStore.
- Enforces authenticated student identity checks across all student-specific operations.
- Read-only queries (get_student_progress, get_pending_objectives, get_current_skill,
  validate_activity_completion) never mutate database state.
- update_student_progress enforces strict RBAC (only assessment_agent/system_admin),
  canonical objective/item validation, prerequisite checks, and duplicate update prevention.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agents.state import (
    ALL_CANONICAL_OBJECTIVE_IDS,
    ALL_CANONICAL_SKILL_IDS,
    AUTHORIZED_PROGRESS_MUTATORS,
    UnauthorizedAccessError,
    UnauthorizedProgressMutationError,
)
from agents.tools.curriculum_tools import get_progress_structure
from agents.tools.registry import ToolValidationError
from learning.interaction_store import LearningStore
from learning.progress_service import ProgressService

logger = logging.getLogger(__name__)

_GLOBAL_LEARNING_STORE: Optional[LearningStore] = None
_GLOBAL_PROGRESS_SERVICE: Optional[ProgressService] = None
_GLOBAL_LEARNING_STORE_GETTER: Optional[Any] = None
_GLOBAL_PROGRESS_SERVICE_GETTER: Optional[Any] = None


def set_default_learning_store(store_or_getter: Any) -> None:
    """Configures the shared LearningStore instance or dynamic getter used across agent tools."""
    global _GLOBAL_LEARNING_STORE, _GLOBAL_LEARNING_STORE_GETTER
    if callable(store_or_getter) and not hasattr(store_or_getter, "get_progress"):
        _GLOBAL_LEARNING_STORE_GETTER = store_or_getter
        _GLOBAL_LEARNING_STORE = None
    else:
        _GLOBAL_LEARNING_STORE = store_or_getter
        _GLOBAL_LEARNING_STORE_GETTER = None


def set_default_progress_service(service_or_getter: Any) -> None:
    """Configures the shared ProgressService instance or dynamic getter used across agent tools."""
    global _GLOBAL_PROGRESS_SERVICE, _GLOBAL_PROGRESS_SERVICE_GETTER
    if callable(service_or_getter) and not hasattr(service_or_getter, "completion_state"):
        _GLOBAL_PROGRESS_SERVICE_GETTER = service_or_getter
        _GLOBAL_PROGRESS_SERVICE = None
    else:
        _GLOBAL_PROGRESS_SERVICE = service_or_getter
        _GLOBAL_PROGRESS_SERVICE_GETTER = None


def get_default_learning_store() -> LearningStore:
    """Provides a shared LearningStore instance."""
    global _GLOBAL_LEARNING_STORE
    if _GLOBAL_LEARNING_STORE_GETTER is not None:
        try:
            store = _GLOBAL_LEARNING_STORE_GETTER()
            if store is not None:
                return store
        except Exception as err:
            logger.warning("Error resolving dynamic learning_store getter: %s", err)
    if _GLOBAL_LEARNING_STORE is None:
        _GLOBAL_LEARNING_STORE = LearningStore()
    return _GLOBAL_LEARNING_STORE


def get_default_progress_service() -> ProgressService:
    """Provides a shared ProgressService instance."""
    global _GLOBAL_PROGRESS_SERVICE
    if _GLOBAL_PROGRESS_SERVICE_GETTER is not None:
        try:
            service = _GLOBAL_PROGRESS_SERVICE_GETTER()
            if service is not None:
                return service
        except Exception as err:
            logger.warning("Error resolving dynamic progress_service getter: %s", err)
    if _GLOBAL_PROGRESS_SERVICE is None:
        _GLOBAL_PROGRESS_SERVICE = ProgressService()
    return _GLOBAL_PROGRESS_SERVICE


def _assert_identity(student_id: str, authenticated_student_id: str) -> None:
    """Verifies that the requested student_id strictly matches the authenticated JWT identity."""
    if not student_id or not isinstance(student_id, str):
        raise UnauthorizedAccessError("student_id must be a valid non-empty string.")
    if not authenticated_student_id or not isinstance(authenticated_student_id, str):
        raise UnauthorizedAccessError("authenticated_student_id must be provided from authenticated session.")

    if student_id.strip() != authenticated_student_id.strip():
        raise UnauthorizedAccessError(
            f"Security violation: Target student_id '{student_id}' does not match "
            f"authenticated identity '{authenticated_student_id}'."
        )


def get_student_progress(
    student_id: str,
    authenticated_student_id: str,
    skill_id: Optional[str] = None,
    learning_store: Optional[LearningStore] = None,
    progress_service: Optional[ProgressService] = None,
) -> Dict[str, Any]:
    """Tool 2: Retrieves student curriculum progress.

    Reuses LearningStore.get_all_progress and ProgressService.calculate.
    Read-only: does not modify any progress.
    """
    _assert_identity(student_id, authenticated_student_id)

    store = learning_store or get_default_learning_store()
    service = progress_service or get_default_progress_service()

    if skill_id is not None:
        normalized_skill = skill_id.strip()
        if normalized_skill not in ALL_CANONICAL_SKILL_IDS:
            raise ToolValidationError(f"Invalid skill_id '{skill_id}'. Must be a canonical SEEK skill.")
        skill_id = normalized_skill

    progress_docs = store.get_all_progress(student_id) or []
    calculated = service.calculate(progress_docs)

    if skill_id:
        skill_summary = next(
            (s for s in calculated.get("skills", []) if s.get("skill_id") == skill_id),
            None,
        )
        if not skill_summary:
            # Default empty summary for this skill
            skill_summary = {
                "skill_id": skill_id,
                "completed": False,
                "progress_percentage": 0,
                "completed_objective_ids": [],
                "completed_item_ids": [],
                "objectives": [],
            }
        return {
            "student_id": student_id,
            "skill": skill_summary,
            "overall_completed_percentage": calculated.get("overall_completed_percentage", 0),
        }

    return {
        "student_id": student_id,
        "overall_completed_percentage": calculated.get("overall_completed_percentage", 0),
        "total_objectives_completed": calculated.get("total_objectives_completed", 0),
        "skills": calculated.get("skills", []),
    }


def get_pending_objectives(
    student_id: str,
    authenticated_student_id: str,
    skill_id: Optional[str] = None,
    learning_store: Optional[LearningStore] = None,
    progress_service: Optional[ProgressService] = None,
) -> List[Dict[str, Any]]:
    """Tool 3: Returns uncompleted canonical objectives for the student.

    Reuses ProgressService.structure and LearningStore.get_all_progress.
    Read-only: prevents hallucinating non-canonical objectives.
    """
    _assert_identity(student_id, authenticated_student_id)

    store = learning_store or get_default_learning_store()
    service = progress_service or get_default_progress_service()

    if skill_id is not None:
        normalized_skill = skill_id.strip()
        if normalized_skill not in ALL_CANONICAL_SKILL_IDS:
            raise ToolValidationError(f"Invalid skill_id '{skill_id}'. Must be a canonical SEEK skill.")
        skill_id = normalized_skill

    progress_docs = store.get_all_progress(student_id) or []
    completed_objectives: set[str] = set()
    for doc in progress_docs:
        completed_objectives.update(doc.get("completed_objective_ids", []))

    structure = getattr(service, "structure", None) or get_progress_structure()
    pending_list = []

    for chapter in structure.get("chapters", []):
        chap_id = chapter.get("chapter_id")
        if skill_id and chap_id != skill_id:
            continue

        for obj in chapter.get("objectives", []):
            obj_id = obj.get("objective_id")
            if obj_id in completed_objectives:
                continue

            pending_list.append({
                "objective_id": obj_id,
                "title": obj.get("title", ""),
                "skill_id": chap_id,
                "required_item_ids": obj.get("required_item_ids", []),
                "is_completed": False,
            })

    return pending_list


def get_current_skill(
    student_id: str,
    authenticated_student_id: str,
    learning_store: Optional[LearningStore] = None,
    progress_service: Optional[ProgressService] = None,
) -> Dict[str, Any]:
    """Tool 4: Determines the student's active empathy skill in curriculum order.

    Reuses ProgressService and LearningStore. Read-only.
    """
    _assert_identity(student_id, authenticated_student_id)

    store = learning_store or get_default_learning_store()
    service = progress_service or get_default_progress_service()

    progress_docs = store.get_all_progress(student_id) or []
    calculated = service.calculate(progress_docs)

    skills_summary = calculated.get("skills", [])

    # Find the first incomplete skill in canonical sequence skill_01 .. skill_08
    active_skill_summary = None
    for skill_info in skills_summary:
        if not skill_info.get("completed"):
            active_skill_summary = skill_info
            break

    # If all skills complete, default to skill_08
    if not active_skill_summary and skills_summary:
        active_skill_summary = skills_summary[-1]

    if not active_skill_summary:
        return {
            "skill_id": "skill_01",
            "title": "Calming the Body and Mind",
            "completed": False,
            "progress_percentage": 0,
            "next_objective_id": "obj_01_01",
        }

    # Find first incomplete objective within the active skill
    objectives = active_skill_summary.get("objectives", [])
    next_obj = next((o.get("objective_id") for o in objectives if not o.get("completed")), None)

    return {
        "skill_id": active_skill_summary.get("skill_id"),
        "title": active_skill_summary.get("title", ""),
        "completed": active_skill_summary.get("completed", False),
        "progress_percentage": active_skill_summary.get("progress_percentage", 0),
        "next_objective_id": next_obj,
    }


def validate_activity_completion(
    student_id: str,
    authenticated_student_id: str,
    skill_id: str,
    progress_item_id: str,
    progress_item_type: str,
    learning_store: Optional[LearningStore] = None,
    progress_service: Optional[ProgressService] = None,
) -> Dict[str, Any]:
    """Tool 7: Validates that an activity exists in the curriculum and matches type.

    Reuses ProgressService.structure and LearningStore.
    Checks skill_id, progress_item_id, progress_item_type, and checks for duplicates.
    Read-only check.
    """
    _assert_identity(student_id, authenticated_student_id)

    if not skill_id or not isinstance(skill_id, str):
        raise ToolValidationError("skill_id is required.")
    if not progress_item_id or not isinstance(progress_item_id, str):
        raise ToolValidationError("progress_item_id is required.")
    if not progress_item_type or not isinstance(progress_item_type, str):
        raise ToolValidationError("progress_item_type is required.")

    norm_skill = skill_id.strip()
    norm_item = progress_item_id.strip()
    norm_type = progress_item_type.strip().lower()
    if norm_type == "practice":
        norm_type = "activity"
    if norm_type == "assessment":
        norm_type = "quiz"

    service = progress_service or get_default_progress_service()
    store = learning_store or get_default_learning_store()
    structure = getattr(service, "structure", None) or get_progress_structure()

    chapter = next((c for c in structure.get("chapters", []) if c.get("chapter_id") == norm_skill), None)
    if not chapter:
        return {
            "is_valid": False,
            "error": f"Skill chapter '{norm_skill}' does not exist in curriculum.",
            "already_completed": False,
        }

    assigned_item = None
    assigned_obj = None
    for obj in chapter.get("objectives", []):
        for item in obj.get("items", []):
            if item.get("item_id") == norm_item:
                assigned_item = item
                assigned_obj = obj
                break
        if assigned_item:
            break

    if not assigned_item:
        return {
            "is_valid": False,
            "error": f"Item '{norm_item}' is not assigned under skill '{norm_skill}'.",
            "already_completed": False,
        }

    assigned_type = str(assigned_item.get("type", "")).lower()
    if assigned_type == "practice":
        assigned_type = "activity"
    if assigned_type == "assessment":
        assigned_type = "quiz"

    if norm_type != assigned_type:
        return {
            "is_valid": False,
            "error": f"item_type mismatch: expected '{assigned_type}', received '{norm_type}'.",
            "already_completed": False,
        }

    # Check whether already completed
    current_progress = store.get_progress(student_id, norm_skill)
    already_completed = norm_item in current_progress.get("completed_item_ids", [])

    return {
        "is_valid": True,
        "skill_id": norm_skill,
        "item_id": norm_item,
        "item_type": assigned_type,
        "objective_id": (assigned_obj or {}).get("objective_id"),
        "already_completed": already_completed,
    }


def update_student_progress(
    student_id: str,
    authenticated_student_id: str,
    authorizing_agent: str,
    skill_id: str,
    item_id: Optional[str] = None,
    objective_id: Optional[str] = None,
    item_type: Optional[str] = None,
    quiz_score: Optional[float] = None,
    source: str = "chatbot",
    verification_result: Optional[Dict[str, Any]] = None,
    learning_store: Optional[LearningStore] = None,
    progress_service: Optional[ProgressService] = None,
) -> Dict[str, Any]:
    """Tool 9: Controlled mutation of student curriculum progress.

    Enforces:
    - Identity check: student_id == authenticated_student_id.
    - RBAC check: authorizing_agent in AUTHORIZED_PROGRESS_MUTATORS.
    - Reuses LearningStore.complete_item and LearningStore.complete_objective.
    - Uses existing curriculum structure validation.
    - Prevents duplicate progress writes.
    """
    _assert_identity(student_id, authenticated_student_id)

    # RBAC Authorization
    normalized_agent = (authorizing_agent or "").strip().lower()
    if normalized_agent not in AUTHORIZED_PROGRESS_MUTATORS:
        raise UnauthorizedProgressMutationError(
            f"Agent role '{authorizing_agent}' is not authorized to update progress. "
            f"Authorized roles are: {sorted(AUTHORIZED_PROGRESS_MUTATORS)}"
        )

    if not skill_id or not isinstance(skill_id, str):
        raise ToolValidationError("skill_id is required.")

    norm_skill = skill_id.strip()
    if norm_skill not in ALL_CANONICAL_SKILL_IDS:
        raise ToolValidationError(f"Invalid skill_id '{skill_id}'. Must be a canonical SEEK skill.")

    store = learning_store or get_default_learning_store()
    service = progress_service or get_default_progress_service()

    current_progress = store.get_progress(student_id, norm_skill)
    mutation_performed = False
    item_already_completed = False
    objective_already_completed = False

    # 1. Mutate item completion if item_id provided
    if item_id:
        norm_item = item_id.strip()
        validation = validate_activity_completion(
            student_id=student_id,
            authenticated_student_id=authenticated_student_id,
            skill_id=norm_skill,
            progress_item_id=norm_item,
            progress_item_type=item_type or "activity",
            learning_store=store,
            progress_service=service,
        )
        if not validation["is_valid"]:
            raise ToolValidationError(validation["error"])

        assigned_obj_id = validation.get("objective_id")
        assigned_type = validation.get("item_type")

        # Resolve requirements for the assigned objective
        structure = getattr(service, "structure", None) or get_progress_structure()
        chapter = next((c for c in structure.get("chapters", []) if c.get("chapter_id") == norm_skill), None)
        assigned_obj_def = next(
            (o for o in (chapter or {}).get("objectives", []) if o.get("objective_id") == assigned_obj_id),
            None,
        )
        item_reqs = (
            ProgressService.completion_requirements(assigned_obj_def)
            if assigned_obj_def
            else {"activity_item_ids": [], "quiz_passing_scores": {}}
        )

        # Duplicate check:
        if validation["already_completed"] and assigned_type != "quiz":
            item_already_completed = True
        else:
            if assigned_type == "quiz":
                if quiz_score is None or not isinstance(quiz_score, (int, float)) or quiz_score < 0 or quiz_score > 100:
                    raise ToolValidationError("Valid quiz_score between 0 and 100 is required for quizzes.")
                store.complete_item(
                    student_id=student_id,
                    skill_id=norm_skill,
                    item_id=norm_item,
                    quiz_score=float(quiz_score),
                    objective_id=assigned_obj_id,
                    objective_completion_requirements=item_reqs,
                    source=source,
                )
            else:
                store.complete_item(
                    student_id=student_id,
                    skill_id=norm_skill,
                    item_id=norm_item,
                    objective_id=assigned_obj_id,
                    objective_completion_requirements=item_reqs,
                    source=source,
                )
            mutation_performed = True

    # 2. Mutate objective completion if objective_id provided
    if objective_id:
        norm_obj = objective_id.strip()
        if norm_obj not in ALL_CANONICAL_OBJECTIVE_IDS:
            raise ToolValidationError(f"Objective '{norm_obj}' is not a canonical SEEK objective.")

        # 2a. Reject reflection submissions from completing objectives
        if verification_result and verification_result.get("submission_type") == "reflection":
            raise UnauthorizedProgressMutationError(
                f"Objective '{norm_obj}' cannot be marked complete from reflection text alone."
            )

        if norm_obj in current_progress.get("completed_objective_ids", []):
            objective_already_completed = True
        else:
            # 2b. Strictly require verification proof - remove generic success=True or raw score >= 90 bypass!
            is_verified = False
            if verification_result and isinstance(verification_result, dict):
                is_verified = verification_result.get("verified") is True

            if not is_verified:
                raise UnauthorizedProgressMutationError(
                    f"Objective '{norm_obj}' cannot be marked complete without verified proof."
                )

            # 2c. Enforce ProgressService completion rules: Activity MUST be completed!
            structure = getattr(service, "structure", None) or get_progress_structure()
            chapter = next(
                (c for c in structure.get("chapters", []) if c.get("chapter_id") == norm_skill),
                None,
            )
            objective_def = next(
                (o for o in (chapter or {}).get("objectives", []) if o.get("objective_id") == norm_obj),
                None,
            )

            reqs = (
                ProgressService.completion_requirements(objective_def)
                if objective_def
                else {"activity_item_ids": []}
            )
            required_activity_ids = reqs.get("activity_item_ids", [])

            completed_item_ids = set(current_progress.get("completed_item_ids", []))
            if item_id and mutation_performed:
                completed_item_ids.add(item_id.strip())

            activity_completed = (
                any(act_id in completed_item_ids for act_id in required_activity_ids)
                if required_activity_ids
                else True
            )

            if not activity_completed:
                raise UnauthorizedProgressMutationError(
                    f"Objective '{norm_obj}' cannot be marked complete because its required activity has not been completed."
                )

            # 2d. If verification is via quiz submission, enforce that the quiz is passed in trusted storage
            if verification_result and verification_result.get("submission_type") == "quiz":
                quiz_scores = dict(current_progress.get("quiz_scores", {}))
                if item_id and assigned_type == "quiz" and quiz_score is not None:
                    quiz_scores[norm_item] = float(quiz_score)

                comp_state = ProgressService.completion_state(
                    objective_def,
                    completed_item_ids,
                    quiz_scores,
                )
                if not comp_state["quiz_passed"]:
                    raise UnauthorizedProgressMutationError(
                        f"Objective '{norm_obj}' cannot be marked complete because its quiz has not been passed in trusted server records."
                    )

            # Construct canonical objective record using actual curriculum structure
            objective_record = {
                "id": norm_obj,
                "title": (objective_def or {}).get("title", ""),
                "content": (objective_def or {}).get("content", ""),
            }

            # Resolve next uncompleted objective within this skill
            completed_for_skill = set(current_progress.get("completed_objective_ids", []))
            completed_for_skill.add(norm_obj)

            chapter_objectives = [
                {
                    "id": o.get("objective_id"),
                    "title": o.get("title", ""),
                    "content": o.get("content", ""),
                }
                for o in (chapter or {}).get("objectives", [])
                if o.get("objective_id")
            ]
            next_objective = next(
                (
                    o for o in chapter_objectives
                    if o["id"] not in completed_for_skill
                ),
                None,
            )

            try:
                store.complete_objective(
                    student_id,
                    norm_skill,
                    objective_record,
                    next_objective,
                )
            except TypeError:
                # Fallback for custom test mocks expecting keyword objective_id
                store.complete_objective(
                    student_id=student_id,
                    skill_id=norm_skill,
                    objective_id=norm_obj,
                )
            mutation_performed = True

    updated_progress = store.get_progress(student_id, norm_skill) or {}
    completed_obj_ids = list(updated_progress.get("completed_objective_ids", []))
    if objective_id and norm_obj not in completed_obj_ids:
        if objective_already_completed or mutation_performed:
            completed_obj_ids.append(norm_obj)

    return {
        "success": True,
        "mutation_performed": mutation_performed,
        "skill_id": norm_skill,
        "item_already_completed": item_already_completed,
        "objective_already_completed": objective_already_completed,
        "completed_item_ids": updated_progress.get("completed_item_ids", []),
        "completed_objective_ids": completed_obj_ids,
    }

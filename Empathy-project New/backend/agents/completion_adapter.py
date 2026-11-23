"""HTTP completion adapter for translating agent assessments into frontend response contracts.

Preserves:
- Exact HTTP 200 progress document contract.
- Exact HTTP 422 prerequisite failure contract.
- Canonical 400 (invalid input), 401 (unauthenticated), 403 (unauthorized student),
  and 503 (database error) status codes.
- Strict identity, prerequisite checking, 90% quiz threshold, and reflection boundaries.
"""

from typing import Any, Dict, Optional, Tuple
from pymongo.errors import PyMongoError

from agents.definitions import AssessmentAgent, AssessmentInput
from agents.state import UnauthorizedAccessError
from agents.tools.curriculum_tools import get_progress_structure
from agents.tools.progress_tools import (
    get_default_learning_store,
    get_default_progress_service,
)
from learning.interaction_store import LearningStore
from learning.progress_service import ProgressService


class ObjectiveCompletionAdapter:
    """Adapts AssessmentAgent execution results into Flask/REST HTTP responses."""

    def __init__(
        self,
        assessment_agent: Optional[AssessmentAgent] = None,
        learning_store: Optional[LearningStore] = None,
        progress_service: Optional[ProgressService] = None,
    ):
        self.assessment_agent = assessment_agent or AssessmentAgent()
        self.learning_store = learning_store
        self.progress_service = progress_service

    def _get_store(self) -> LearningStore:
        return self.learning_store or get_default_learning_store()

    def _get_service(self) -> ProgressService:
        return self.progress_service or get_default_progress_service()

    def handle_completion_request(
        self,
        student_id: Optional[str],
        authenticated_student_id: Optional[str],
        skill_id: Optional[str],
        objective_id: Optional[str],
        body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], int]:
        """Translates completion requests into exact HTTP responses expected by frontend.

        Returns:
            Tuple[Dict[str, Any], int]: (response_dict, status_code)
        """
        # 1. Authentication check
        if not authenticated_student_id:
            return {"error": "Missing or invalid authorization token."}, 401

        # 2. Student identity assertion
        if not student_id or str(student_id).strip() != str(authenticated_student_id).strip():
            return {
                "error": "Security violation: Target student_id does not match authenticated identity."
            }, 403

        # 3. Input validation
        body = body or {}
        norm_skill = (skill_id or "").strip()
        norm_obj = (objective_id or "").strip()

        if not norm_skill or not norm_obj:
            return {"error": "A valid skill_id and learning objective are required."}, 400

        service = self._get_service()
        store = self._get_store()
        structure = getattr(service, "structure", None) or get_progress_structure()

        chapter = next(
            (c for c in structure.get("chapters", []) if c.get("chapter_id") == norm_skill),
            None,
        )
        objective_def = next(
            (o for o in (chapter or {}).get("objectives", []) if o.get("objective_id") == norm_obj),
            None,
        )

        if not chapter or not objective_def:
            return {"error": "A valid skill_id and learning objective are required."}, 400

        # 4. Storage operations guarded against PyMongoError
        try:
            current = store.get_progress(student_id, norm_skill)
        except PyMongoError:
            return {"error": "Could not update student progress."}, 503
        except Exception as err:
            if "PyMongoError" in type(err).__name__:
                return {"error": "Could not update student progress."}, 503
            return {"error": "Could not update student progress."}, 503

        # Idempotency check: If objective is already marked complete, preserve existing progress
        if norm_obj in current.get("completed_objective_ids", []):
            return current, 200

        # 5. Extract submission details
        source = body.get("source", "skills_page")
        submission_type = body.get("submission_type")
        if not submission_type:
            submission_type = "quick_check" if source == "chatbot" else "quiz"
        else:
            submission_type = str(submission_type).strip().lower()

        interaction_id = body.get("interaction_id")
        item_id = body.get("item_id")
        item_type = body.get("item_type")

        # Resolve student answer based on submission type
        if submission_type == "quick_check":
            student_answer = body.get("quick_check_answer")
            if student_answer is None:
                student_answer = body.get("student_answer")
        elif submission_type == "reflection":
            student_answer = body.get("student_answer") or body.get("reflection_text") or ""
        elif submission_type == "quiz":
            client_score = body.get("quiz_score") if body.get("quiz_score") is not None else body.get("student_answer")
            if client_score is not None:
                student_answer = client_score
            else:
                persisted_scores = current.get("quiz_scores", {})
                student_answer = max(persisted_scores.values()) if persisted_scores else 0.0
        else:
            student_answer = body.get("student_answer")

        # 6. Execute evaluation and progress assessment through AssessmentAgent
        try:
            output = self.assessment_agent.execute(
                AssessmentInput(
                    student_id=student_id,
                    authenticated_student_id=authenticated_student_id,
                    submission_type=submission_type,
                    student_answer=student_answer,
                    objective_id=norm_obj,
                    skill_id=norm_skill,
                    item_id=item_id,
                    item_type=item_type,
                    interaction_id=interaction_id,
                    commit_progress=True,
                )
            )
        except UnauthorizedAccessError as err:
            return {"error": str(err)}, 403
        except PyMongoError:
            return {"error": "Could not update student progress."}, 503
        except Exception as err:
            if "PyMongoError" in type(err).__name__:
                return {"error": "Could not update student progress."}, 503
            return {"error": f"Evaluation error: {err}"}, 400

        # 7. Formulate response based on objective_completed outcome
        if output.objective_completed:
            try:
                persisted = store.get_progress(student_id, norm_skill)
            except PyMongoError:
                return {"error": "Could not update student progress."}, 503

            return persisted, 200

        # 8. Unsuccessful completion: formulate exact HTTP 422 contract
        comp_state = ProgressService.completion_state(
            objective_def,
            current.get("completed_item_ids", []),
            current.get("quiz_scores", {}),
        )

        missing_conditions = []
        if not comp_state["activity_completed"]:
            missing_conditions.append("activity")
        if not comp_state["quiz_passed"] and not (output.verified and submission_type == "quick_check"):
            missing_conditions.append("quiz")

        if source == "chatbot" and not output.verified:
            completion_error = "A correct, verified understanding check is required to complete this objective."
        elif output.verified and not comp_state["activity_completed"]:
            completion_error = "Complete the objective activity before completing this objective."
        elif submission_type == "reflection":
            completion_error = "Objective cannot be marked complete from reflection text alone."
        elif comp_state["activity_completed"] and not comp_state["quiz_passed"]:
            completion_error = "Complete the objective activity and pass its quiz before completing this objective."
        else:
            completion_error = "Complete the objective activity and pass its quiz before completing this objective."

        failure_payload = {
            "error": completion_error,
            "completed": False,
            "objective_id": norm_obj,
            "required_item_ids": comp_state["required_item_ids"],
            "completed_item_ids": sorted(
                set(current.get("completed_item_ids", []))
                & set(comp_state["required_item_ids"])
            ),
            "missing_item_ids": comp_state["missing_required_item_ids"],
            "missing_completion_conditions": missing_conditions,
        }

        return failure_payload, 422

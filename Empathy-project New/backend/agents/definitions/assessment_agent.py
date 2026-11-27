"""AssessmentAgent implementation for Empathy Guru.

Evaluates student answers and reflections, handles approved learning activities,
verifies understanding, checks objective coverage, and safely commits validated
activity completion through controlled tools.
Never marks an activity completed without valid confirmation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from agents.definitions.base_agent import AgentExecutionError, BaseAgent
from agents.state import (
    ALL_CANONICAL_OBJECTIVE_IDS,
    AgentState,
    LearningStage,
    UnauthorizedProgressMutationError,
)
from agents.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class AssessmentInput:
    """Structured input for AssessmentAgent."""
    student_id: str
    authenticated_student_id: str
    submission_type: str  # "quick_check" | "reflection" | "quiz"
    student_answer: Any
    objective_id: str
    skill_id: str
    item_id: Optional[str] = None
    item_type: Optional[str] = None
    interaction_id: Optional[str] = None
    commit_progress: bool = True


@dataclass
class AssessmentOutput:
    """Structured output for AssessmentAgent."""
    verified: bool
    score: Optional[float]
    feedback: str
    objective_completed: bool
    progress_mutation_applied: bool
    item_id: Optional[str] = None
    objective_id: str = ""
    updated_progress: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verified": self.verified,
            "score": self.score,
            "feedback": self.feedback,
            "objective_completed": self.objective_completed,
            "progress_mutation_applied": self.progress_mutation_applied,
            "item_id": self.item_id,
            "objective_id": self.objective_id,
            "updated_progress": self.updated_progress,
            "metadata": self.metadata,
        }


class AssessmentAgent(BaseAgent):
    """Specialized assessment, verification, and progress commitment agent."""

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        timeout_seconds: float = 15.0,
    ):
        super().__init__(
            role="assessment_agent",
            tool_registry=tool_registry,
            timeout_seconds=timeout_seconds,
        )

    def execute(
        self,
        input_data: AssessmentInput,
        state: Optional[AgentState] = None,
    ) -> AssessmentOutput:
        """Executes verification and authorized progress update."""
        if not isinstance(input_data, AssessmentInput):
            raise AgentExecutionError("input_data must be an instance of AssessmentInput.")

        if input_data.student_id != input_data.authenticated_student_id:
            raise AgentExecutionError("Security violation: student_id does not match authenticated identity.")

        if input_data.objective_id not in ALL_CANONICAL_OBJECTIVE_IDS:
            raise AgentExecutionError(f"Objective '{input_data.objective_id}' is not a canonical SEEK objective.")

        # Update state stage if provided
        if state:
            try:
                state.transition_to(LearningStage.ASSESSMENT, reason="AssessmentAgent evaluating response")
            except Exception:
                state.current_learning_stage = LearningStage.ASSESSMENT

        # 1. Evaluate student response using controlled tool
        evaluation = self.call_tool(
            "evaluate_student_response",
            student_id=input_data.student_id,
            authenticated_student_id=input_data.authenticated_student_id,
            objective_id=input_data.objective_id,
            submission_type=input_data.submission_type,
            student_answer=input_data.student_answer,
            skill_id=input_data.skill_id,
            interaction_id=input_data.interaction_id,
            item_id=input_data.item_id,
        )

        is_verified = evaluation.get("verified", False)
        score = evaluation.get("score")
        feedback = evaluation.get("feedback", "")
        progress_applied = False
        updated_progress: Optional[Dict[str, Any]] = None

        # 2. Only if verified and commit_progress is requested, coordinate progress update via authorized tool
        if is_verified and input_data.commit_progress:
            # Reflection submissions must NEVER complete objectives
            if input_data.submission_type == "reflection":
                target_objective_id = None
            else:
                target_objective_id = input_data.objective_id

            try:
                progress_result = self.call_tool(
                    "update_student_progress",
                    student_id=input_data.student_id,
                    authenticated_student_id=input_data.authenticated_student_id,
                    authorizing_agent=self.role,
                    skill_id=input_data.skill_id,
                    item_id=input_data.item_id,
                    objective_id=target_objective_id,
                    item_type=input_data.item_type or "activity",
                    quiz_score=None,  # Never trust client-supplied scores for quiz completion!
                    verification_result=evaluation,
                )
                progress_applied = progress_result.get("mutation_performed", False)
                updated_progress = progress_result

                # Update in-memory AgentState if supplied and objective actually completed
                if (
                    state
                    and target_objective_id
                    and (
                        progress_result.get("objective_already_completed")
                        or (
                            progress_applied
                            and target_objective_id in progress_result.get("completed_objective_ids", [])
                        )
                    )
                ):
                    state.record_objective_completion(
                        objective_id=target_objective_id,
                        authorizing_agent=self.role,
                        verification_result=evaluation,
                    )
            except UnauthorizedProgressMutationError as err:
                logger.warning("AssessmentAgent progress update not authorized: %s", err)
                progress_applied = False
                feedback = f"{feedback} (Objective not completed: {err})".strip()
            except Exception as err:
                logger.error("Failed to commit progress via tool: %s", err)
                progress_applied = False
                feedback = f"{feedback} (Note: Progress update could not be saved: {err})".strip()

        # Determine if objective is actually completed in verified progress
        objective_completed = False
        if (
            input_data.submission_type != "reflection"
            and updated_progress
            and isinstance(updated_progress, dict)
        ):
            completed_objs = updated_progress.get("completed_objective_ids", [])
            objective_completed = input_data.objective_id in completed_objs

        output = AssessmentOutput(
            verified=is_verified,
            score=score,
            feedback=feedback,
            objective_completed=objective_completed,
            progress_mutation_applied=progress_applied,
            item_id=input_data.item_id,
            objective_id=input_data.objective_id,
            updated_progress=updated_progress,
            metadata={
                "submission_type": input_data.submission_type,
            },
        )

        return output

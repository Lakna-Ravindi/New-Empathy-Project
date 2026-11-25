"""LearningPlannerAgent implementation for Empathy Guru.

Reads existing student progress, identifies completed and pending objectives,
selects the next appropriate objective following pedagogical rules, and recommends
the next learning action. Never independently updates progress.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.definitions.base_agent import AgentExecutionError, BaseAgent
from agents.prompts.learning_planner_prompts import (
    LEARNING_PLANNER_SYSTEM_INSTRUCTION,
    build_planner_prompt,
)
from agents.state import (
    ALL_CANONICAL_OBJECTIVE_IDS,
    ALL_CANONICAL_SKILL_IDS,
    AgentState,
    LearningStage,
)
from knowledge.objective_mapper import CANONICAL_OBJECTIVE_IDS
from agents.tools.curriculum_tools import (
    OBJECTIVE_CANONICAL_MAP,
    get_content_skills,
    get_keyword_skill_map,
    get_knowledge_base,
    get_progress_structure,
)
from agents.tools.registry import ToolRegistry
from learning.pedagogical_controller import (
    belongs_to_skill,
    curriculum_skill_id,
    detect_emotion,
    get_terms,
    process_student_question,
)
from learning.semantic_search import semantic_search

logger = logging.getLogger(__name__)


@dataclass
class LearningPlannerInput:
    """Structured input for LearningPlannerAgent."""
    student_id: str
    authenticated_student_id: str
    student_question: str
    current_skill_id: Optional[str] = None
    current_objective_id: Optional[str] = None
    completed_objectives: Optional[List[str]] = None


@dataclass
class LearningPlannerOutput:
    """Structured output for LearningPlannerAgent."""
    skill: Dict[str, Any]
    learning_objective: Dict[str, Any]
    recommended_activity: Optional[Dict[str, Any]]
    relevant_chunks: List[Dict[str, Any]]
    recommended_next_action: str
    detected_emotion: str
    pending_objectives_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill": self.skill,
            "learning_objective": self.learning_objective,
            "recommended_activity": self.recommended_activity,
            "relevant_chunks": self.relevant_chunks,
            "recommended_next_action": self.recommended_next_action,
            "detected_emotion": self.detected_emotion,
            "pending_objectives_count": self.pending_objectives_count,
            "metadata": self.metadata,
        }


class LearningPlannerAgent(BaseAgent):
    """Specialized learning planning and pedagogical sequencing agent."""

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        timeout_seconds: float = 15.0,
    ):
        super().__init__(
            role="learning_planner_agent",
            tool_registry=tool_registry,
            timeout_seconds=timeout_seconds,
        )

    def execute(
        self,
        input_data: LearningPlannerInput,
        state: Optional[AgentState] = None,
    ) -> LearningPlannerOutput:
        """Executes curriculum planning and objective selection."""
        if not isinstance(input_data, LearningPlannerInput):
            raise AgentExecutionError("input_data must be an instance of LearningPlannerInput.")

        if input_data.student_id != input_data.authenticated_student_id:
            raise AgentExecutionError("Security violation: student_id does not match authenticated identity.")

        question = (input_data.student_question or "").strip()
        if not question:
            raise AgentExecutionError("student_question cannot be empty.")

        # Update state stage if provided
        if state:
            try:
                state.transition_to(LearningStage.LEARNING_PLANNING, reason="LearningPlannerAgent planning path")
            except Exception:
                state.current_learning_stage = LearningStage.LEARNING_PLANNING

        # 1. Read student progress and pending objectives using controlled tools
        student_progress = self.call_tool(
            "get_student_progress",
            student_id=input_data.student_id,
            authenticated_student_id=input_data.authenticated_student_id,
        )

        pending_objectives = self.call_tool(
            "get_pending_objectives",
            student_id=input_data.student_id,
            authenticated_student_id=input_data.authenticated_student_id,
        )

        active_skill_info = self.call_tool(
            "get_current_skill",
            student_id=input_data.student_id,
            authenticated_student_id=input_data.authenticated_student_id,
        )

        # 2. Inspect student question before selecting a learning objective
        detected_emotion = detect_emotion(question)

        matched_skill_id: Optional[str] = None
        matched_objective_id: Optional[str] = None
        matched_skill_title: Optional[str] = None
        activity: Optional[Dict[str, Any]] = None

        try:
            kb = get_knowledge_base()
            prog_struct = get_progress_structure()
            km = get_keyword_skill_map()
            decision = process_student_question(
                question,
                kb,
                highlighted_keywords=km,
                progress_structure=prog_struct,
            )
            if decision and decision.get("status") == "learning_path_selected":
                skill_info = decision.get("skill") or {}
                obj_info = decision.get("learning_objective") or {}

                s_id = skill_info.get("skill_id") or skill_info.get("id")
                o_id = obj_info.get("objective_id") or obj_info.get("id")

                if s_id and s_id not in ALL_CANONICAL_SKILL_IDS:
                    s_id = skill_info.get("skill_id")
                if s_id in ALL_CANONICAL_SKILL_IDS:
                    matched_skill_id = s_id
                    matched_skill_title = skill_info.get("title", "")
                if o_id in ALL_CANONICAL_OBJECTIVE_IDS:
                    canonical_o_id = OBJECTIVE_CANONICAL_MAP.get(o_id, o_id)
                    matched_objective_id = canonical_o_id
                activity = decision.get("recommended_activity")
        except Exception as proc_err:
            logger.debug("process_student_question in planner encountered an issue: %s", proc_err)

        # Validate decision against student question concepts and curriculum evidence
        normalized_q = question.lower().replace("-", " ")
        if matched_skill_id:
            if (
                "empathy" in normalized_q
                and "self compassion" not in normalized_q
                and matched_skill_id != "skill_07"
            ):
                logger.warning(
                    "Planner detected skill mismatch: empathy question assigned to %s; correcting to skill_07.",
                    matched_skill_id,
                )
                matched_skill_id = "skill_07"
                matched_objective_id = "obj_07_01"
                matched_skill_title = "Skill 7: Empathic Concern"
                activity = None
            elif (
                ("self compassion" in normalized_q or "self-compassion" in normalized_q)
                and matched_skill_id != "skill_04"
            ):
                logger.warning(
                    "Planner detected skill mismatch: self-compassion question assigned to %s; correcting to skill_04.",
                    matched_skill_id,
                )
                matched_skill_id = "skill_04"
                matched_objective_id = "obj_04_04"
                matched_skill_title = "Skill 4: Self-Compassion"
                activity = None

        if not matched_skill_id or not matched_objective_id:
            try:
                kb_chunks = get_knowledge_base()
                evidence = semantic_search(question, kb_chunks, limit=3)
                evidence_skills = [
                    c.get("skill_id") for c in evidence
                    if c.get("skill_id") in ALL_CANONICAL_SKILL_IDS
                ]
                if evidence_skills:
                    cand_skill = evidence_skills[0]
                    cand_obj = next(
                        (
                            c.get("objective_id") for c in evidence
                            if c.get("skill_id") == cand_skill
                            and c.get("objective_id") in ALL_CANONICAL_OBJECTIVE_IDS
                        ),
                        None,
                    )
                    if cand_skill and cand_obj:
                        matched_skill_id = cand_skill
                        matched_objective_id = cand_obj
            except Exception as sem_err:
                logger.debug("Planner semantic evidence check issue: %s", sem_err)

        # 3. Determine target skill and objective
        # If reliable topic match exists, select it for answering the question;
        # otherwise preserve existing progress-based planning behavior.
        if matched_skill_id and matched_objective_id:
            target_skill_id = matched_skill_id
            target_objective_id = matched_objective_id
            target_skill_title = matched_skill_title or ""
        else:
            target_skill_id = input_data.current_skill_id or active_skill_info.get("skill_id", "skill_01")
            target_skill_title = active_skill_info.get("title", "")
            target_objective_id = None
            if pending_objectives:
                skill_pending = [o for o in pending_objectives if o.get("skill_id") == target_skill_id]
                if skill_pending:
                    target_objective_id = skill_pending[0].get("objective_id")
                else:
                    target_objective_id = pending_objectives[0].get("objective_id")
                    target_skill_id = pending_objectives[0].get("skill_id", target_skill_id)

            if not target_objective_id:
                target_objective_id = "obj_01_01"
                target_skill_id = "skill_01"

        # Ensure target objective belongs canonically to target skill
        if target_skill_id in CANONICAL_OBJECTIVE_IDS:
            valid_objs = set(CANONICAL_OBJECTIVE_IDS[target_skill_id].values())
            if target_objective_id not in valid_objs:
                target_objective_id = next(iter(valid_objs), "obj_01_01")

        target_objective_title = ""
        try:
            for s in get_content_skills():
                if s.get("skill_id") == target_skill_id:
                    if not target_skill_title:
                        target_skill_title = s.get("skill_title", "")
                    for o in s.get("objectives", []):
                        if o.get("objective_id") == target_objective_id:
                            target_objective_title = o.get("objective_title", "")
                            break
                    break
        except Exception:
            pass

        if not target_skill_title:
            target_skill_title = active_skill_info.get("title", "")

        # 4. Retrieve relevant SEEK chunks and approved activity via controlled tools
        relevant_chunks = self.call_tool(
            "retrieve_approved_seek_content",
            query=question,
            skill_id=target_skill_id,
            objective_id=target_objective_id,
            limit=5,
        )

        if not relevant_chunks:
            # Fallback to unconstrained retrieval within approved curriculum
            relevant_chunks = self.call_tool(
                "retrieve_approved_seek_content",
                query=question,
                limit=5,
            )

        if not activity:
            try:
                activity = self.call_tool(
                    "get_learning_activity",
                    skill_id=target_skill_id,
                    objective_id=target_objective_id,
                )
            except Exception as err:
                logger.debug("Planner did not locate specific activity: %s", err)

        recommended_action = (
            "proceed_to_activity" if activity else "introduce_concept_and_check"
        )

        output = LearningPlannerOutput(
            skill={"id": target_skill_id, "title": target_skill_title},
            learning_objective={
                "id": target_objective_id,
                "objective_id": target_objective_id,
                "title": target_objective_title,
            },
            recommended_activity=activity,
            relevant_chunks=relevant_chunks,
            recommended_next_action=recommended_action,
            detected_emotion=detected_emotion,
            pending_objectives_count=len(pending_objectives),
            metadata={
                "overall_completed_percentage": student_progress.get("overall_completed_percentage", 0),
            },
        )

        if state:
            state.current_skill_id = target_skill_id
            state.current_objective_id = target_objective_id
            state.recommended_next_action = {"action": recommended_action}
            state.metadata["recommended_activity"] = activity
            state.metadata["detected_terms"] = sorted(get_terms(question))

        return output

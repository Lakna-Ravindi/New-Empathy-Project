"""TutorAgent implementation for Empathy Guru.

Answers questions using approved SEEK Knowledge Base content, explains concepts clearly,
connects concepts to everyday student-life examples, and strictly enforces the 4-stage contract.
Never invents curriculum content.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.definitions.base_agent import AgentExecutionError, BaseAgent
from agents.prompts.tutor_prompts import TUTOR_SYSTEM_INSTRUCTION, build_tutor_prompt
from agents.state import AgentState, LearningStage, StateValidationError
from agents.tools.registry import ToolRegistry
from learning.gemini_responder import (
    _deduplicate_content,
    _fallback_response,
    _parse_response,
    _valid_activity_metadata,
)

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import ClientError, ServerError
except ImportError:  # pragma: no cover
    genai = None
    types = None
    ClientError = Exception
    ServerError = Exception

logger = logging.getLogger(__name__)


@dataclass
class TutorInput:
    """Structured input for TutorAgent."""
    student_id: str
    authenticated_student_id: str
    student_question: str
    skill_id: str
    objective_id: str
    activity: Optional[Dict[str, Any]] = None
    relevant_chunks: Optional[List[Dict[str, Any]]] = None
    answer_style: str = "short"  # "short" | "long"
    support_intro: Optional[str] = None


@dataclass
class TutorOutput:
    """Structured output for TutorAgent adhering to the 4-stage contract."""
    answer: str
    suggested_questions: List[str]
    steps: List[Dict[str, Any]]
    skill_id: str
    objective_id: str
    is_fallback: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "suggested_questions": self.suggested_questions,
            "steps": self.steps,
            "skill_id": self.skill_id,
            "objective_id": self.objective_id,
            "is_fallback": self.is_fallback,
            "metadata": self.metadata,
        }


class TutorAgent(BaseAgent):
    """Specialized instructional agent synthesizing approved SEEK content."""

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        timeout_seconds: float = 15.0,
    ):
        super().__init__(
            role="tutor_agent",
            tool_registry=tool_registry,
            timeout_seconds=timeout_seconds,
        )

    def execute(
        self,
        input_data: TutorInput,
        state: Optional[AgentState] = None,
    ) -> TutorOutput:
        """Executes grounded tutorial generation."""
        if not isinstance(input_data, TutorInput):
            raise AgentExecutionError("input_data must be an instance of TutorInput.")

        if input_data.student_id != input_data.authenticated_student_id:
            raise AgentExecutionError("Security violation: student_id does not match authenticated identity.")

        if not input_data.student_question.strip():
            raise AgentExecutionError("student_question cannot be empty.")

        # Update state stage if provided
        if state:
            try:
                state.transition_to(LearningStage.TUTORING, reason="TutorAgent generating instruction")
            except Exception:
                state.current_learning_stage = LearningStage.TUTORING

        # 1. Fetch relevant SEEK chunks through controlled tool if not provided
        relevant_chunks = input_data.relevant_chunks
        if not relevant_chunks:
            try:
                relevant_chunks = self.call_tool(
                    "retrieve_approved_seek_content",
                    query=input_data.student_question,
                    skill_id=input_data.skill_id,
                    objective_id=input_data.objective_id,
                    limit=5,
                )
            except Exception as err:
                logger.warning("Controlled content retrieval failed: %s", err)
                relevant_chunks = []

        # 2. Fetch approved activity through controlled tool if not provided
        activity = input_data.activity
        if not activity:
            try:
                activity = self.call_tool(
                    "get_learning_activity",
                    skill_id=input_data.skill_id,
                    objective_id=input_data.objective_id,
                )
            except Exception as err:
                logger.debug("No specific activity retrieved via tool: %s", err)
                activity = None

        # Build learning context required for Gemini responder
        learning_context = {
            "student_question": input_data.student_question,
            "status": "learning_path_selected",
            "skill": {"id": input_data.skill_id},
            "learning_objective": {"id": input_data.objective_id, "objective_id": input_data.objective_id},
            "recommended_activity": activity,
            "question_relevant_content": relevant_chunks,
        }

        # 3. Call Gemini API using existing integration or fallback
        api_key = os.getenv("GEMINI_API_KEY")
        parsed_payload: Optional[Dict[str, Any]] = None
        is_fallback = False

        if api_key and genai is not None:
            model_name = os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
            prompt_str = build_tutor_prompt(
                student_question=input_data.student_question,
                selected_skill=learning_context["skill"],
                selected_objective=learning_context["learning_objective"],
                selected_activity=activity,
                relevant_chunks=relevant_chunks,
                answer_style=input_data.answer_style,
                support_intro=input_data.support_intro,
            )

            client = genai.Client(api_key=api_key)
            for attempt in range(2):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt_str,
                        config=types.GenerateContentConfig(
                            system_instruction=TUTOR_SYSTEM_INSTRUCTION,
                            max_output_tokens=2048,
                            response_mime_type="application/json",
                        ),
                    )
                    if response and response.text:
                        parsed_payload = _parse_response(
                            response.text,
                            selected_activity=activity,
                            allow_empty_suggestions=True,
                        )
                        break
                except Exception as err:
                    logger.warning("Gemini generation attempt %d failed: %s", attempt + 1, err)
                    time.sleep(0.5)

        # 4. If Gemini failed or was unavailable, use deterministic fallback engine
        if not parsed_payload:
            logger.info("Using deterministic educational fallback response.")
            parsed_payload = _fallback_response(learning_context, input_data.answer_style)
            is_fallback = True

        # Preserve emotional support introduction for dual-intent routes without duplication
        if input_data.support_intro and isinstance(input_data.support_intro, str):
            intro = input_data.support_intro.strip()
            if intro:
                current_answer = str(parsed_payload.get("answer") or "").strip()
                if intro.lower() not in current_answer.lower():
                    parsed_payload["answer"] = f"{intro} {current_answer}" if current_answer else intro

                steps = parsed_payload.get("steps") or []
                if steps and isinstance(steps, list) and isinstance(steps[0], dict):
                    step_content = str(steps[0].get("content") or "").strip()
                    if intro.lower() not in step_content.lower():
                        steps[0]["content"] = f"{intro} {step_content}" if step_content else intro

        output = TutorOutput(
            answer=parsed_payload["answer"],
            suggested_questions=parsed_payload.get("suggested_questions", []),
            steps=parsed_payload.get("steps", []),
            skill_id=input_data.skill_id,
            objective_id=input_data.objective_id,
            is_fallback=is_fallback,
            metadata={
                "activity_included": activity is not None,
                "chunks_count": len(relevant_chunks),
            },
        )

        if state:
            state.set_response(output.to_dict())

        return output

"""EmpathyOrchestrator: Central coordinator for the UNESCO MGIEP SEEK multi-agent system.

Coordinates:
- AgentRouter: Intent and emotion classification
- AgentStateManager: Isolated session and turn memory
- Domain Agents: EmotionalSupportAgent, LearningPlannerAgent, TutorAgent, AssessmentAgent
- ResponseValidator: Contract and schema enforcement
- Persistence: Seamless reuse of existing LearningStore and collections
- Guaranteed Fallback: Invokes deterministic pedagogical fallback when Gemini is unavailable
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

from pymongo.errors import PyMongoError

from agents.definitions import (
    AssessmentAgent,
    AssessmentInput,
    AssessmentOutput,
    EmotionalSupportAgent,
    EmotionalSupportInput,
    EmotionalSupportOutput,
    GeneralEmpathyAgent,
    GeneralEmpathyInput,
    GeneralEmpathyOutput,
    LearningPlannerAgent,
    LearningPlannerInput,
    LearningPlannerOutput,
    TutorAgent,
    TutorInput,
    TutorOutput,
)
from agents.definitions.tutor_agent import _already_contains_acknowledgment
from agents.router import AgentRouter, StudentIntent
from agents.state import (
    AgentState,
    AgentStateManager,
    LearningStage,
    UnauthorizedAccessError,
)
from agents.tools import ToolRegistry, get_default_tool_registry
from agents.validator import AgentResponseValidator
from learning.interaction_store import LearningStore
from learning.pedagogical_controller import answer_student_question
from learning.progress_service import ProgressService

logger = logging.getLogger(__name__)


class EmpathyOrchestrator:
    """Coordinates execution across specialized domain agents while preserving existing contracts."""

    def __init__(
        self,
        learning_store: Optional[LearningStore] = None,
        progress_service: Optional[ProgressService] = None,
        tool_registry: Optional[ToolRegistry] = None,
        state_manager: Optional[AgentStateManager] = None,
        attach_metadata_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
        fallback_answer_fn: Optional[Callable] = None,
        override_check_fn: Optional[Callable[[], bool]] = None,
        learning_store_getter: Optional[Callable[[], Any]] = None,
        progress_service_getter: Optional[Callable[[], Any]] = None,
        knowledge_base: Optional[Any] = None,
        keyword_skill_map: Optional[Any] = None,
        general_empathy_agent: Optional[GeneralEmpathyAgent] = None,
    ):
        self._learning_store = learning_store or LearningStore()
        self._progress_service = progress_service or ProgressService()
        self.tool_registry = tool_registry or get_default_tool_registry()
        self.state_manager = state_manager or AgentStateManager()
        self.attach_metadata_fn = attach_metadata_fn
        self.fallback_answer_fn = fallback_answer_fn or answer_student_question
        self.override_check_fn = override_check_fn
        self.learning_store_getter = learning_store_getter
        self.progress_service_getter = progress_service_getter
        self.knowledge_base = knowledge_base
        self.keyword_skill_map = keyword_skill_map

        # Initialize domain agents
        self.tutor_agent = TutorAgent(tool_registry=self.tool_registry)
        self.support_agent = EmotionalSupportAgent(tool_registry=self.tool_registry)
        self.general_empathy_agent = general_empathy_agent or GeneralEmpathyAgent(
            tool_registry=self.tool_registry,
            fallback_on_error=True,
        )
        self.planner_agent = LearningPlannerAgent(tool_registry=self.tool_registry)
        self.assessment_agent = AssessmentAgent(tool_registry=self.tool_registry)

        # Ensure agent tools share the same LearningStore and ProgressService instances
        from agents.tools.progress_tools import (
            set_default_learning_store,
            set_default_progress_service,
        )
        set_default_learning_store(lambda: self.learning_store)
        set_default_progress_service(lambda: self.progress_service)

    @property
    def learning_store(self) -> Any:
        if self.learning_store_getter:
            return self.learning_store_getter()
        return self._learning_store

    @property
    def progress_service(self) -> Any:
        if self.progress_service_getter:
            return self.progress_service_getter()
        return self._progress_service

    def orchestrate_turn(
        self,
        student_id: str,
        authenticated_student_id: str,
        message: str,
        answer_style: str = "short",
        conversation_id: str = "default",
        request_id: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], int]:
        """Orchestrates an educational or emotional dialogue turn.

        Returns:
            Tuple of (response_payload: Dict[str, Any], status_code: int)
        """
        # 1. Identity assertion
        if not student_id or not authenticated_student_id or student_id.strip() != authenticated_student_id.strip():
            return {
                "error": "Security violation: Authenticated student ID does not match request ID."
            }, 403

        question = (message or "").strip()
        if not question:
            return {"error": "Please enter a question."}, 400

        if answer_style not in {"short", "long"}:
            return {"error": "Invalid answer style."}, 400

        # Check for explicit mock override
        if self.override_check_fn and self.override_check_fn():
            return self._execute_fallback(student_id, question, answer_style)

        # 2. Get or create isolated state
        state = self.state_manager.get_or_create_state(
            authenticated_student_id=authenticated_student_id,
            conversation_id=conversation_id,
            answer_style=answer_style,
        )
        state.current_message = question

        # Populate history if provided from request context and state has empty history
        if history and isinstance(history, list) and not state.conversation_history:
            for item in history:
                if isinstance(item, dict):
                    role = item.get("role") or item.get("sender") or ("user" if item.get("isUser") else "assistant")
                    content = (
                        item.get("content")
                        or item.get("message")
                        or item.get("response")
                        or item.get("text")
                        or ""
                    )
                    if str(content).strip():
                        state.add_turn_to_history(str(role).lower(), str(content).strip())

        # Record user turn into unified conversational history (avoiding duplicate user turns)
        if not state.conversation_history or not (
            state.conversation_history[-1].get("role") == "user"
            and state.conversation_history[-1].get("content") == question
        ):
            state.add_turn_to_history("user", question)

        # Reset completed stage if starting a new turn
        if state.current_learning_stage == LearningStage.COMPLETED:
            try:
                state.transition_to(LearningStage.INITIALIZED, reason="New turn started")
            except Exception:
                state.current_learning_stage = LearningStage.INITIALIZED

        # 3. Intent Classification
        routing_decision = AgentRouter.classify_intent(question, state=state)
        intent = routing_decision["intent"]
        detected_emotion = routing_decision["detected_emotion"]
        route_target = routing_decision["route_target"]
        state.detected_intent = intent

        # Branch A: Greeting
        if intent == StudentIntent.GREETING.value:
            greeting_text = (
                "Hey! 👋 I'm here to help you explore empathy and build your skills. "
                "What would you like to learn about today?"
            )
            try:
                state.transition_to(LearningStage.COMPLETED, reason="Greeting responded")
            except Exception:
                state.current_learning_stage = LearningStage.COMPLETED

            greeting_payload = {
                "educational_response": greeting_text,
                "answer": greeting_text,
                "response": greeting_text,
                "active_agent": "system",
                "role": "system",
                "learning_context": {
                    "message": greeting_text,
                    "student_question": question,
                    "detected_emotion": "calm",
                    "status": "greeting",
                    "skill": None,
                    "topic": None,
                    "answer_style": answer_style,
                },
                "steps": [],
                "suggested_questions": [],
            }
            if not state.conversation_history or not (
                state.conversation_history[-1].get("role") == "assistant"
                and state.conversation_history[-1].get("content") == greeting_text
            ):
                state.add_turn_to_history("assistant", greeting_text, extra={"active_agent": "system"})
            return greeting_payload, 200

        # Branch B: Pure Emotional Distress / Venting / General Empathy
        if route_target == "emotional_support_agent":
            try:
                try:
                    if state.current_learning_stage in {LearningStage.INITIALIZED, LearningStage.INTENT_ROUTING}:
                        state.transition_to(LearningStage.EMOTIONAL_SUPPORT, reason="Emotional support selected")
                except Exception:
                    state.current_learning_stage = LearningStage.EMOTIONAL_SUPPORT

                support_message = None
                role_name = "general_empathy_agent"
                active_agent = "general_empathy_agent"

                # 1. Execute GeneralEmpathyAgent if available
                if self.general_empathy_agent is not None:
                    try:
                        ge_out: GeneralEmpathyOutput = self.general_empathy_agent.execute(
                            GeneralEmpathyInput(
                                student_id=student_id,
                                authenticated_student_id=authenticated_student_id,
                                current_message=question,
                                conversation_history=state.conversation_history,
                                request_id=request_id,
                                fallback_on_error=True,
                            ),
                            state=state,
                        )
                        support_message = ge_out.response
                        role_name = ge_out.role or "general_empathy_agent"
                    except Exception as ge_err:
                        logger.warning(
                            "GeneralEmpathyAgent execution failed, falling back to EmotionalSupportAgent: %s",
                            ge_err,
                        )

                # 2. Fallback to EmotionalSupportAgent if needed
                if not support_message:
                    support_out: EmotionalSupportOutput = self.support_agent.execute(
                        EmotionalSupportInput(
                            student_id=student_id,
                            authenticated_student_id=authenticated_student_id,
                            current_message=question,
                            conversation_history=state.conversation_history,
                            emotional_context=state.emotional_context,
                        ),
                        state=state,
                    )
                    support_message = support_out.support_message
                    role_name = "emotional_support_agent"

                response_dict = {
                    "educational_response": support_message,
                    "answer": support_message,
                    "response": support_message,
                    "active_agent": active_agent,
                    "role": role_name,
                    "learning_context": {
                        "student_question": question,
                        "detected_emotion": detected_emotion,
                        "status": "emotional_support",
                        "skill": None,
                        "topic": None,
                        "answer_style": answer_style,
                        "message": support_message,
                    },
                    "steps": [],
                    "suggested_questions": [],
                }
                # Save interaction for conversational continuity (does NOT mutate student_progress)
                try:
                    interaction_id = self.learning_store.save_interaction(
                        student_id=student_id,
                        learning_context=response_dict["learning_context"],
                        educational_response=support_message,
                        learning_steps=[],
                    )
                    response_dict["interaction_id"] = interaction_id
                except PyMongoError:
                    return {"error": "Could not save the learning interaction."}, 503
                except Exception as save_err:
                    logger.warning("Could not persist emotional interaction: %s", save_err)

                if not state.conversation_history or not (
                    state.conversation_history[-1].get("role") == "assistant"
                    and state.conversation_history[-1].get("content") == support_message
                ):
                    state.add_turn_to_history("assistant", support_message, extra={"active_agent": active_agent})
                try:
                    state.transition_to(LearningStage.COMPLETED, reason="Emotional support turn completed")
                except Exception:
                    state.current_learning_stage = LearningStage.COMPLETED

                return response_dict, 200

            except Exception as err:
                logger.error("Emotional support routing error: %s", err, exc_info=True)

        # Branch C: Curriculum Planning & Tutoring Instruction
        support_intro: Optional[str] = None
        if route_target == "support_then_tutor":
            try:
                support_out = self.support_agent.execute(
                    EmotionalSupportInput(
                        student_id=student_id,
                        authenticated_student_id=authenticated_student_id,
                        current_message=question,
                        conversation_history=state.conversation_history,
                        emotional_context=state.emotional_context,
                    ),
                    state=state,
                )
                support_intro = support_out.empathetic_acknowledgment
            except Exception as err:
                logger.warning("Support intro generation failed: %s", err)

        try:
            # 4a. Planning
            planner_out: LearningPlannerOutput = self.planner_agent.execute(
                LearningPlannerInput(
                    student_id=student_id,
                    authenticated_student_id=authenticated_student_id,
                    student_question=question,
                    current_skill_id=state.current_skill_id,
                ),
                state=state,
            )

            # 4b. Tutoring
            tutor_out: TutorOutput = self.tutor_agent.execute(
                TutorInput(
                    student_id=student_id,
                    authenticated_student_id=authenticated_student_id,
                    student_question=question,
                    skill_id=planner_out.skill["id"],
                    objective_id=planner_out.learning_objective["id"],
                    activity=planner_out.recommended_activity,
                    relevant_chunks=planner_out.relevant_chunks,
                    answer_style=answer_style,
                    support_intro=support_intro,
                ),
                state=state,
            )

            # 4c. Response formatting and validation
            response_payload = tutor_out.to_dict()
            rec_activity = planner_out.recommended_activity or {}

            learning_context = {
                "student_question": question,
                "detected_emotion": planner_out.detected_emotion,
                "detected_terms": list(state.metadata.get("detected_terms", [])),
                "status": "learning_path_selected",
                "skill": planner_out.skill,
                "learning_objective": planner_out.learning_objective,
                "recommended_activity": rec_activity,
                "answer_style": answer_style,
            }

            # Attach progress metadata
            if self.attach_metadata_fn:
                self.attach_metadata_fn(learning_context)
            AgentResponseValidator.attach_progress_metadata(response_payload, learning_context.get("recommended_activity"))

            response_payload["learning_context"] = learning_context
            response_payload["educational_response"] = response_payload.get("answer", "")
            response_payload["response"] = response_payload["educational_response"]
            response_payload["active_agent"] = "tutor_agent"
            response_payload["role"] = "tutor_agent"

            # 4d. Persist interaction to MongoDB
            try:
                interaction_id = self.learning_store.save_interaction(
                    student_id=student_id,
                    learning_context=learning_context,
                    educational_response=response_payload["educational_response"],
                    learning_steps=response_payload.get("steps", []),
                )
                response_payload["interaction_id"] = interaction_id

                # Record learning event if activity present
                final_rec_activity = learning_context.get("recommended_activity") or {}
                progress_item_id = final_rec_activity.get("progress_item_id")
                if progress_item_id and planner_out.skill.get("id"):
                    self.learning_store.record_learning_event(
                        student_id=student_id,
                        skill_id=planner_out.skill["id"],
                        item_id=progress_item_id,
                        objective_id=final_rec_activity.get("progress_objective_id"),
                        source="chatbot",
                        status="introduced",
                    )
            except PyMongoError:
                return {"error": "Could not save the learning interaction."}, 503
            except Exception as store_err:
                logger.error("Could not persist learning interaction: %s", store_err)
                return {"error": "Could not save the learning interaction."}, 503

            assistant_content = str(response_payload.get("educational_response", ""))
            if not state.conversation_history or not (
                state.conversation_history[-1].get("role") == "assistant"
                and state.conversation_history[-1].get("content") == assistant_content
            ):
                state.add_turn_to_history(
                    "assistant",
                    assistant_content,
                    extra={"active_agent": "tutor_agent"},
                )
            return response_payload, 200

        except Exception as err:
            logger.error("Agent orchestration encountered error, executing fallback: %s", err, exc_info=True)
            return self._execute_fallback(student_id, question, answer_style, support_intro=support_intro)

    def _execute_fallback(
        self,
        student_id: str,
        question: str,
        answer_style: str,
        support_intro: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], int]:
        """Executes the preserved pedagogical fallback engine without failing the request."""
        try:
            progress_docs = self.learning_store.get_all_progress(student_id)
        except PyMongoError:
            progress_docs = None

        kb = self.knowledge_base
        if kb is None:
            try:
                kb = self.tool_registry.get_tool("retrieve_approved_seek_content").__globals__.get("get_knowledge_base", lambda: [])()
            except Exception:
                kb = []

        result = self.fallback_answer_fn(
            question,
            kb,
            self.keyword_skill_map or {},
            answer_style,
            progress_documents=progress_docs,
            progress_structure=self.progress_service.get_structure(),
        )

        learning_context = result.get("learning_context", result)
        if isinstance(learning_context, dict):
            learning_context.setdefault("student_question", question)
            learning_context.setdefault("status", "learning_path_selected")
        result.setdefault("suggested_questions", [])

        if self.attach_metadata_fn:
            self.attach_metadata_fn(learning_context)

        recommended_activity = learning_context.get("recommended_activity") or {}
        AgentResponseValidator.attach_progress_metadata(result, recommended_activity)
        steps = result.get("steps", [])

        educational_response = result.get(
            "educational_response",
            result.get("answer", steps if steps else learning_context.get("message", "")),
        )

        result["active_agent"] = "tutor_agent"
        result["role"] = "tutor_agent"
        result["response"] = educational_response if isinstance(educational_response, str) else result.get("answer", "")

        if support_intro and isinstance(support_intro, str):
            intro = support_intro.strip()
            if intro:
                if isinstance(educational_response, str) and not _already_contains_acknowledgment(educational_response, intro):
                    educational_response = f"{intro} {educational_response}" if educational_response else intro
                    result["educational_response"] = educational_response
                    result["answer"] = educational_response
                if steps and isinstance(steps, list) and isinstance(steps[0], dict):
                    step_content = str(steps[0].get("content") or "").strip()
                    if not _already_contains_acknowledgment(step_content, intro):
                        steps[0]["content"] = f"{intro} {step_content}" if step_content else intro

        try:
            interaction_id = self.learning_store.save_interaction(
                student_id,
                learning_context,
                educational_response,
                learning_steps=steps,
            )
            result["interaction_id"] = interaction_id

            progress_item_id = recommended_activity.get("progress_item_id")
            if progress_item_id and learning_context.get("skill"):
                skill_id = learning_context["skill"].get("id") or learning_context["skill"].get("skill_id")
                if skill_id:
                    self.learning_store.record_learning_event(
                        student_id,
                        skill_id,
                        progress_item_id,
                        objective_id=recommended_activity.get("progress_objective_id"),
                        source="chatbot",
                        status="introduced",
                    )
            return result, 200

        except PyMongoError:
            return {"error": "Could not save the learning interaction."}, 503

    def handle_objective_completion(
        self,
        student_id: str,
        authenticated_student_id: str,
        skill_id: str,
        objective_id: str,
        quick_check_answer: Optional[Any] = None,
        interaction_id: Optional[str] = None,
        source: str = "chatbot",
        submission_type: str = "quick_check",
        student_answer: Optional[Any] = None,
        item_id: Optional[str] = None,
        item_type: Optional[str] = None,
        conversation_id: Optional[str] = None,
        commit_progress: bool = True,
    ) -> AssessmentOutput:
        """Handles assessment and objective verification through AssessmentAgent."""
        actual_answer = student_answer if student_answer is not None else quick_check_answer
        state = None
        if conversation_id:
            state = self.state_manager.get_state(authenticated_student_id, conversation_id)

        return self.assessment_agent.execute(
            AssessmentInput(
                student_id=student_id,
                authenticated_student_id=authenticated_student_id,
                submission_type=submission_type,
                student_answer=actual_answer,
                objective_id=objective_id,
                skill_id=skill_id,
                item_id=item_id,
                item_type=item_type,
                interaction_id=interaction_id,
                commit_progress=commit_progress,
            ),
            state=state,
        )

    def complete_objective_http(
        self,
        student_id: Optional[str],
        authenticated_student_id: Optional[str],
        skill_id: Optional[str],
        objective_id: Optional[str],
        body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], int]:
        """HTTP-compatible entrypoint adapting AssessmentAgent outputs to frontend contract."""
        from agents.completion_adapter import ObjectiveCompletionAdapter

        adapter = ObjectiveCompletionAdapter(
            assessment_agent=self.assessment_agent,
            learning_store=self.learning_store,
            progress_service=self.progress_service,
        )
        return adapter.handle_completion_request(
            student_id=student_id,
            authenticated_student_id=authenticated_student_id,
            skill_id=skill_id,
            objective_id=objective_id,
            body=body,
        )

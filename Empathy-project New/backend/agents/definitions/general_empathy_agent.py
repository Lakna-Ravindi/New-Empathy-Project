"""GeneralEmpathyAgent implementation for Empathy Guru.

Provides compassionate, non-judgmental emotional support, active listening,
and emotional acknowledgment without providing clinical/medical diagnoses,
unsolicited advice, or curriculum instruction.

Strictly separated from the UNESCO MGIEP SEEK curriculum learning system.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.definitions.base_agent import AgentExecutionError, BaseAgent
from agents.prompts.general_empathy_prompts import (
    GENERAL_EMPATHY_SYSTEM_INSTRUCTION,
    build_general_empathy_prompt,
    is_no_advice_requested,
)
from agents.state import AgentState
from agents.tools.registry import ToolRegistry

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError, ClientError, ServerError
except ImportError:  # pragma: no cover
    genai = None
    types = None
    APIError = Exception
    ClientError = Exception
    ServerError = Exception

logger = logging.getLogger(__name__)


@dataclass
class GeneralEmpathyInput:
    """Structured input contract for GeneralEmpathyAgent."""
    student_id: str
    authenticated_student_id: str
    current_message: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    request_id: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    fallback_on_error: Optional[bool] = None


@dataclass
class GeneralEmpathyOutput:
    """Structured output contract for GeneralEmpathyAgent."""
    response: str
    role: str = "general_empathy_agent"
    no_advice_acknowledged: bool = False
    is_fallback: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "response": self.response,
            "role": self.role,
            "no_advice_acknowledged": self.no_advice_acknowledged,
            "is_fallback": self.is_fallback,
            "metadata": self.metadata,
        }

    def to_frontend_dict(self) -> Dict[str, Any]:
        """Provides 100% backward compatibility with the frontend expected shape."""
        return {"response": self.response}


class GeneralEmpathyAgent(BaseAgent):
    """Independent General Empathy Agent providing non-didactic emotional companionship."""

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        timeout_seconds: float = 15.0,
        client: Optional[Any] = None,
        fallback_on_error: bool = True,
    ):
        super().__init__(
            role="general_empathy_agent",
            tool_registry=tool_registry,
            timeout_seconds=timeout_seconds,
        )
        self._client = client
        self.fallback_on_error = fallback_on_error

    def execute(
        self,
        input_data: GeneralEmpathyInput,
        state: Optional[AgentState] = None,
    ) -> GeneralEmpathyOutput:
        """Executes empathetic response generation.

        Raises:
            AgentExecutionError: On validation failure or unrecoverable error when
                fallback is disabled.
        """
        # 1. Type validation
        if not isinstance(input_data, GeneralEmpathyInput):
            raise AgentExecutionError("input_data must be an instance of GeneralEmpathyInput.")

        # 2. Identity security assertion
        if not input_data.student_id or not input_data.authenticated_student_id:
            raise AgentExecutionError("Security violation: Missing student ID or authentication.")
        if input_data.student_id.strip() != input_data.authenticated_student_id.strip():
            raise AgentExecutionError("Security violation: student_id does not match authenticated identity.")

        # 3. Message validation
        message = (input_data.current_message or "").strip()
        if not message:
            raise AgentExecutionError("current_message cannot be empty.")
        if len(message) > 4000:
            raise AgentExecutionError("Messages must be 4000 characters or fewer.")

        # 4. History validation and sanitization
        history = input_data.conversation_history
        if not isinstance(history, list):
            raise AgentExecutionError("conversation_history must be a list.")
        if len(history) > 20:
            raise AgentExecutionError("conversation_history must contain at most 20 messages.")

        sanitized_history: List[Dict[str, Any]] = []
        for idx, entry in enumerate(history):
            if not isinstance(entry, dict):
                raise AgentExecutionError(f"History entry at index {idx} must be a dictionary.")
            role = entry.get("role")
            if role not in {"user", "assistant", "model"}:
                raise AgentExecutionError(f"Invalid role '{role}' in history entry at index {idx}.")
            content = entry.get("content")
            if not isinstance(content, str) or not content.strip():
                raise AgentExecutionError(f"Content in history entry at index {idx} must be non-empty string.")
            if len(content) > 4000:
                raise AgentExecutionError(f"History entry at index {idx} exceeds 4000 characters.")
            sanitized_history.append({"role": role, "content": content.strip()})

        # 5. Preferences & No-Advice detection
        prefs = dict(input_data.preferences or {})
        no_advice = bool(prefs.get("no_advice_requested")) or is_no_advice_requested(message)
        prefs["no_advice_requested"] = no_advice

        # 6. Build prompt turns
        prompt_contents = build_general_empathy_prompt(
            message=message,
            history=sanitized_history,
            preferences=prefs,
        )

        should_fallback = (
            input_data.fallback_on_error
            if input_data.fallback_on_error is not None
            else self.fallback_on_error
        )

        # 7. Gemini Generation
        response_text: Optional[str] = None
        is_fallback = False
        api_key = os.getenv("GEMINI_API_KEY")
        model_name = os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
        timeout_ms = int(os.getenv("GEMINI_TIMEOUT_MS", str(int(self.timeout_seconds * 1000))))

        client = self._client
        if client is None and api_key and genai is not None:
            try:
                http_options = types.HttpOptions(timeout=timeout_ms) if types and hasattr(types, "HttpOptions") else None
                client = genai.Client(api_key=api_key, http_options=http_options)
            except Exception as client_err:
                logger.warning("Could not instantiate Gemini client: %s", client_err)

        if client is not None:
            for attempt in range(2):
                try:
                    config = types.GenerateContentConfig(
                        system_instruction=GENERAL_EMPATHY_SYSTEM_INSTRUCTION,
                        max_output_tokens=1024,
                        temperature=0.7,
                    ) if types and hasattr(types, "GenerateContentConfig") else None

                    gen_kwargs = {
                        "model": model_name,
                        "contents": prompt_contents,
                    }
                    if config is not None:
                        gen_kwargs["config"] = config

                    result = client.models.generate_content(**gen_kwargs)
                    if result and getattr(result, "text", None):
                        candidate_text = str(result.text).strip()
                        if candidate_text:
                            response_text = candidate_text
                            break
                    logger.warning("Gemini returned empty response on attempt %d", attempt + 1)
                except Exception as err:
                    # Sanitize logging: never log api_key
                    error_type = type(err).__name__
                    logger.warning("Gemini generation attempt %d failed (%s): %s", attempt + 1, error_type, err)
                    if attempt == 0:
                        time.sleep(0.5)

        # 8. Fallback or Error escalation
        if not response_text:
            if not should_fallback:
                raise AgentExecutionError("Gemini service temporarily unavailable.")

            is_fallback = True
            lowered = message.lower()
            if no_advice:
                response_text = (
                    "I hear you, and you don't have to figure anything out right now. "
                    "I'm here to listen. Take all the time and space you need."
                )
            elif "exam" in lowered or "study" in lowered or "assignment" in lowered:
                response_text = (
                    "Exam and academic pressures can feel truly exhausting when expectations build up. "
                    "It is completely natural to feel overwhelmed. How are you holding up right now?"
                )
            else:
                response_text = (
                    "Thank you for sharing that with me. It takes courage to open up about how you're feeling, "
                    "and I'm right here with you. What feels like the heaviest part of this today?"
                )

        metadata = {
            "model": model_name,
            "request_id": input_data.request_id,
            "history_turns_count": len(sanitized_history),
        }

        output = GeneralEmpathyOutput(
            response=response_text,
            role=self.role,
            no_advice_acknowledged=no_advice,
            is_fallback=is_fallback,
            metadata=metadata,
        )

        if state is not None:
            state.set_response(output.to_dict())

        return output

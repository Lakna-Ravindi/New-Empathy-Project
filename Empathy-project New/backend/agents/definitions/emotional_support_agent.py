"""EmotionalSupportAgent implementation for Empathy Guru.

Provides compassionate, non-judgmental emotional support, active listening,
and emotional awareness without providing clinical/medical diagnoses or unsolicited advice.
Never overrides the safety system.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.definitions.base_agent import AgentExecutionError, BaseAgent
from agents.prompts.emotional_support_prompts import (
    EMOTIONAL_SUPPORT_SYSTEM_INSTRUCTION,
    build_emotional_support_prompt,
)
from agents.state import (
    AgentState,
    LearningStage,
    validate_conversational_emotional_context,
)
from agents.tools.registry import ToolRegistry

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover
    genai = None
    types = None

logger = logging.getLogger(__name__)

_NO_ADVICE_PATTERNS = re.compile(
    r"\b(don't|dont|do not)\s+(want|need|give me)\s+advice|"
    r"\b(just|only)\s+(want to|wanna)?\s*(listen|vent|talk)|"
    r"\b(stop giving advice|no advice)\b",
    re.IGNORECASE,
)

_LEARNING_READINESS_PATTERNS = re.compile(
    r"\b(ready to learn|feel better now|calmer now|what should i do|"
    r"how do i manage|how to practice|teach me|can we learn)\b",
    re.IGNORECASE,
)


@dataclass
class EmotionalSupportInput:
    """Structured input for EmotionalSupportAgent."""
    student_id: str
    authenticated_student_id: str
    current_message: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    emotional_context: Optional[Dict[str, Any]] = None


@dataclass
class EmotionalSupportOutput:
    """Structured output for EmotionalSupportAgent."""
    empathetic_acknowledgment: str
    gentle_transition_offer: str
    readiness_for_learning: bool
    support_message: str
    emotional_context: Dict[str, Any]
    is_fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "empathetic_acknowledgment": self.empathetic_acknowledgment,
            "gentle_transition_offer": self.gentle_transition_offer,
            "readiness_for_learning": self.readiness_for_learning,
            "support_message": self.support_message,
            "emotional_context": self.emotional_context,
            "is_fallback": self.is_fallback,
        }


class EmotionalSupportAgent(BaseAgent):
    """Specialized emotional support and de-escalation agent."""

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        timeout_seconds: float = 15.0,
    ):
        super().__init__(
            role="emotional_support_agent",
            tool_registry=tool_registry,
            timeout_seconds=timeout_seconds,
        )

    def execute(
        self,
        input_data: EmotionalSupportInput,
        state: Optional[AgentState] = None,
    ) -> EmotionalSupportOutput:
        """Executes compassionate emotional support response."""
        if not isinstance(input_data, EmotionalSupportInput):
            raise AgentExecutionError("input_data must be an instance of EmotionalSupportInput.")

        if input_data.student_id != input_data.authenticated_student_id:
            raise AgentExecutionError("Security violation: student_id does not match authenticated identity.")

        message = (input_data.current_message or "").strip()
        if not message:
            raise AgentExecutionError("current_message cannot be empty.")

        # Update state stage if provided
        if state:
            try:
                state.transition_to(LearningStage.EMOTIONAL_SUPPORT, reason="EmotionalSupportAgent active")
            except Exception:
                state.current_learning_stage = LearningStage.EMOTIONAL_SUPPORT

        # Validate emotional context (strictly enforces no clinical diagnoses)
        raw_context = input_data.emotional_context or {
            "emotion": "reflective",
            "valence": "neutral",
            "intensity": "moderate",
            "topic_focus": "personal",
        }
        validated_context = validate_conversational_emotional_context(raw_context)

        # Detect specific preferences and readiness
        no_advice = bool(_NO_ADVICE_PATTERNS.search(message))
        readiness = bool(_LEARNING_READINESS_PATTERNS.search(message))
        validated_context["readiness_for_learning"] = readiness
        validated_context["no_advice_requested"] = no_advice

        api_key = os.getenv("GEMINI_API_KEY")
        support_text: Optional[str] = None
        is_fallback = False

        if api_key and genai is not None:
            prompt_content = build_emotional_support_prompt(
                student_message=message,
                emotional_context=validated_context,
                conversation_history=input_data.conversation_history,
                no_advice_requested=no_advice,
            )

            client = genai.Client(api_key=api_key)
            try:
                response = client.models.generate_content(
                    model=os.getenv("GEMINI_MODEL") or "gemini-2.5-flash",
                    contents=prompt_content,
                    config=types.GenerateContentConfig(
                        system_instruction=EMOTIONAL_SUPPORT_SYSTEM_INSTRUCTION,
                        max_output_tokens=512,
                        temperature=0.7,
                    ),
                )
                if response and response.text and response.text.strip():
                    support_text = response.text.strip()
            except Exception as err:
                logger.warning("Emotional support Gemini call failed: %s", err)

        # Fallback if Gemini unavailable or failed
        if not support_text:
            is_fallback = True
            if no_advice:
                support_text = (
                    "I hear you completely, and you don't have to figure anything out right now. "
                    "I'm here to listen. Take all the time and space you need."
                )
            elif "exam" in message.lower() or "study" in message.lower():
                support_text = (
                    "Exam stress can feel truly exhausting when expectations build up. "
                    "It is completely natural to feel overwhelmed. How are you holding up right now?"
                )
            else:
                support_text = (
                    "Thank you for sharing that with me. It takes courage to open up about how you're feeling, "
                    "and I'm right here with you. What feels like the heaviest part of this today?"
                )

        acknowledgment = support_text.split(".")[0].strip() + "."
        transition_offer = (
            "Whenever you feel ready, we can look at a gentle exercise to help steady your mind."
            if readiness
            else ""
        )

        output = EmotionalSupportOutput(
            empathetic_acknowledgment=acknowledgment,
            gentle_transition_offer=transition_offer,
            readiness_for_learning=readiness,
            support_message=support_text,
            emotional_context=validated_context,
            is_fallback=is_fallback,
        )

        if state:
            state.update_emotional_context(validated_context)
            state.set_response({
                "educational_response": support_text,
                "answer": support_text,
                "steps": [],
                "suggested_questions": [],
            })

        return output

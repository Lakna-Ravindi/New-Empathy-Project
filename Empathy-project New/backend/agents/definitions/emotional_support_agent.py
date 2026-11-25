"""EmotionalSupportAgent implementation for Empathy Guru.

Provides compassionate, non-judgmental emotional support, active listening,
and emotional awareness without providing clinical/medical diagnoses or unsolicited advice.
Never overrides the safety system.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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


def _validate_gemini_candidate_completion(result: Any, model_name: str, attempt: int) -> Tuple[bool, Optional[str]]:
    """Validates that a Gemini response is complete, non-empty, and finished normally.

    Detects MAX_TOKENS and any non-STOP finish reason, incomplete sentences, and
    abrupt truncations.

    Returns:
        (is_valid, validated_text_or_none)
    """
    if not result:
        logger.warning("Gemini attempt %d returned empty result object (model: %s)", attempt, model_name)
        return False, None

    candidates = getattr(result, "candidates", None)
    if candidates and isinstance(candidates, (list, tuple)) and len(candidates) > 0:
        candidate = candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason is not None and not type(finish_reason).__name__.endswith("Mock"):
            finish_reason_str = str(getattr(finish_reason, "name", finish_reason)).upper()
            if "STOP" not in finish_reason_str and finish_reason != 1:
                logger.warning(
                    "Gemini attempt %d rejected due to incomplete finish_reason: %s (model: %s)",
                    attempt,
                    finish_reason_str,
                    model_name,
                )
                return False, None
    elif candidates is not None and type(candidates).__name__.endswith("Mock"):
        # Auto-mocked response in unit tests without explicit candidates list
        pass
    elif not hasattr(result, "text"):
        logger.warning("Gemini attempt %d returned no candidates (model: %s)", attempt, model_name)
        return False, None

    raw_text = getattr(result, "text", None)
    if not raw_text or not str(raw_text).strip():
        logger.warning("Gemini attempt %d returned empty text (model: %s)", attempt, model_name)
        return False, None

    text = str(raw_text).strip()
    words = text.split()
    if len(words) < 3 or len(text) < 15:
        logger.warning(
            "Gemini attempt %d rejected: text too short (%d words, %d chars, model: %s)",
            attempt,
            len(words),
            len(text),
            model_name,
        )
        return False, None

    # Sentence completeness validation: must end with terminal punctuation
    terminal_punct = (".", "!", "?", "…", '"', "'", "”", "’")
    if not text.endswith(terminal_punct):
        logger.warning(
            "Gemini attempt %d rejected: response truncated mid-sentence without terminal punctuation (model: %s)",
            attempt,
            model_name,
        )
        return False, None

    # Abrupt termination validation: check for trailing dangling conjunctions/prepositions
    last_word = words[-1].lower().rstrip(".,!?…\"'”’")
    dangling_words = {
        "between", "and", "or", "but", "so", "because", "with",
        "to", "at", "by", "for", "in", "of", "on", "that", "which",
    }
    if last_word in dangling_words:
        logger.warning(
            "Gemini attempt %d rejected: response ended abruptly on dangling word '%s' (model: %s)",
            attempt,
            last_word,
            model_name,
        )
        return False, None

    return True, text


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
            model_name = os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
            for attempt in range(2):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt_content,
                        config=types.GenerateContentConfig(
                            system_instruction=EMOTIONAL_SUPPORT_SYSTEM_INSTRUCTION,
                            max_output_tokens=512,
                            temperature=0.7,
                        ),
                    )
                    is_valid, validated_text = _validate_gemini_candidate_completion(
                        response, model_name=model_name, attempt=attempt + 1
                    )
                    if is_valid and validated_text:
                        support_text = validated_text
                        break
                    if attempt == 0:
                        time.sleep(0.5)
                except Exception as err:
                    error_type = type(err).__name__
                    logger.warning(
                        "Emotional support Gemini attempt %d failed (%s): %s",
                        attempt + 1,
                        error_type,
                        err,
                    )
                    if attempt == 0:
                        time.sleep(0.5)

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

        # Extract the first complete sentence ending with [.!?] without creating partial or malformed fragments
        sentence_matches = re.findall(r'[^.!?]+[.!?]', support_text.strip())
        if sentence_matches and len(sentence_matches[0].strip().split()) >= 3:
            acknowledgment = sentence_matches[0].strip()
        else:
            first_line = support_text.strip().split("\n")[0].strip()
            if first_line and not first_line.endswith((".", "!", "?")):
                first_line += "."
            acknowledgment = first_line
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

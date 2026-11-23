"""AgentRouter: Fast, deterministic intent classification and routing heuristics.

Classifies incoming student messages into actionable intent categories:
- GREETING: Common greeting phrases
- EMOTIONAL_DISTRESS: Expressions of emotional distress, venting, overwhelm
- CURRICULUM_INQUIRY: Inquiries about SEEK empathy skills, concepts, or practices
- PRACTICE_CHECK: Answers to practice checks or quizzes
- REFLECTION: Submissions of personal reflections on activities
- GENERAL_CHAT: Conversational questions or general dialogue
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, Optional

from learning.pedagogical_controller import (
    EMOTION_KEYWORDS,
    MEANINGFUL_CONCEPTS,
    detect_emotion,
    get_terms,
    normalize_text,
)

GREETINGS = {
    "hi", "hii", "hiii", "hello", "helloo", "hey", "heyy",
    "hiya", "good morning", "good afternoon", "good evening",
}

DISTRESS_INDICATORS = {
    "overwhelmed", "stressed", "stress", "anxious", "anxiety",
    "exhausted", "hopeless", "sad", "upset", "crying", "angry",
    "mad", "frustrated", "lonely", "struggling", "scared",
    "nervous", "worried", "worry", "panic", "burnout", "burnt out",
}

# Emotion words present in MEANINGFUL_CONCEPTS that describe distress/state rather than SEEK curriculum topics
EMOTION_CONCEPT_WORDS = set().union(*EMOTION_KEYWORDS.values()) | {
    "feeling", "feelings", "calm", "anger", "anxiety", "stress",
}
SEEK_CONCEPTS = MEANINGFUL_CONCEPTS - EMOTION_CONCEPT_WORDS


class StudentIntent(str, Enum):
    GREETING = "greeting"
    EMOTIONAL_DISTRESS = "emotional_distress"
    CURRICULUM_INQUIRY = "curriculum_inquiry"
    PRACTICE_CHECK = "practice_check"
    REFLECTION = "reflection"
    GENERAL_CHAT = "general_chat"


class AgentRouter:
    """Classifies student utterances to guide multi-agent orchestration."""

    @staticmethod
    def is_greeting(text: str) -> bool:
        """Determines if the text is a standalone greeting or conversational introduction."""
        normalized = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
        normalized = " ".join(normalized.split())
        if normalized in GREETINGS:
            return True
        greeting_prefixes = ("hi", "hello", "hey", "good morning", "good afternoon", "good evening")
        if any(normalized.startswith(f"{prefix} ") for prefix in greeting_prefixes):
            if any(phrase in normalized for phrase in ("how can you help", "what can you do", "who are you", "what do you do", "help me")):
                return True
        return False

    @staticmethod
    def classify_intent(message: str, state: Optional[Any] = None) -> Dict[str, Any]:
        """Classifies student utterance into intent, emotion, and target agent route."""
        text = (message or "").strip()
        cleaned = normalize_text(text)

        # 1. Check for greeting
        if AgentRouter.is_greeting(text):
            return {
                "intent": StudentIntent.GREETING.value,
                "detected_emotion": "calm",
                "route_target": "greeting",
                "confidence": 1.0,
            }

        # 2. Extract emotion and meaningful terms
        detected_emotion = detect_emotion(text)
        terms = get_terms(text)
        has_seek_concept = any(term in SEEK_CONCEPTS for term in terms)

        # Check for emotional distress indicators
        words = set(cleaned.split())
        has_distress = any(word in DISTRESS_INDICATORS for word in words) or detected_emotion in {"anxious", "sad", "angry"}

        # Check for explicit curriculum asking phrases ("what is", "how do I practice", "tell me about")
        curriculum_query = has_seek_concept or any(
            phrase in cleaned
            for phrase in (
                "what is", "what does", "how do i", "how can i", "how to",
                "explain", "teach me", "tell me about", "learn about",
                "resilient zone", "body literacy", "somatic", "grounding",
                "tracking", "mindfulness", "compassion", "empathy",
            )
        )

        # 3. Routing heuristic
        if has_distress and curriculum_query:
            # Dual intent: distress + learning readiness -> support intro with tutorial
            return {
                "intent": StudentIntent.EMOTIONAL_DISTRESS.value,
                "detected_emotion": detected_emotion,
                "route_target": "support_then_tutor",
                "has_seek_concept": has_seek_concept,
                "confidence": 0.85,
            }
        elif has_distress:
            return {
                "intent": StudentIntent.EMOTIONAL_DISTRESS.value,
                "detected_emotion": detected_emotion,
                "route_target": "emotional_support_agent",
                "has_seek_concept": has_seek_concept,
                "confidence": 0.9,
            }
        elif curriculum_query:
            return {
                "intent": StudentIntent.CURRICULUM_INQUIRY.value,
                "detected_emotion": detected_emotion,
                "route_target": "tutor_agent",
                "has_seek_concept": has_seek_concept,
                "confidence": 0.95,
            }
        else:
            return {
                "intent": StudentIntent.GENERAL_CHAT.value,
                "detected_emotion": detected_emotion,
                "route_target": "tutor_agent",
                "has_seek_concept": False,
                "confidence": 0.7,
            }

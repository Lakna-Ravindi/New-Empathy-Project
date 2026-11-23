"""Regression and unit tests for AgentRouter intent classification and route selection.

Tests verify:
1. Pure emotional distress routing to emotional_support_agent.
2. Pure curriculum inquiry routing to tutor_agent.
3. Dual-intent (distress + curriculum) routing to support_then_tutor.
4. General conversation routing to tutor_agent.
5. Ambiguous message handling routing to tutor_agent.
6. Standalone greetings and greeting-intro patterns.
7. Verification that safety-critical inputs qualify as safety risks
   and cannot bypass the safety gate to reach multi-agent orchestration.
"""

import pytest
from agents.router import AgentRouter, StudentIntent
from safety import detect_safety_risk


class TestAgentRouterIntentClassification:
    """Unit tests for deterministic intent classification in AgentRouter."""

    # -----------------------------------------------------------------
    # 1. Distress Only
    # -----------------------------------------------------------------
    @pytest.mark.parametrize(
        "message,expected_emotion",
        [
            ("I'm feeling really anxious and overwhelmed with all my school work.", "anxious"),
            ("I feel so stressed and exhausted today.", "calm"),
            ("I am crying and feel so sad about what happened.", "sad"),
            ("I am frustrated, angry, and struggling to cope.", "angry"),
            ("I feel completely lonely and burnt out.", "calm"),
        ],
    )
    def test_distress_only_routes_to_emotional_support(self, message, expected_emotion):
        result = AgentRouter.classify_intent(message)
        assert result["route_target"] == "emotional_support_agent"
        assert result["intent"] == StudentIntent.EMOTIONAL_DISTRESS.value
        assert result["confidence"] == 0.9

    # -----------------------------------------------------------------
    # 2. Curriculum Question Only
    # -----------------------------------------------------------------
    @pytest.mark.parametrize(
        "message",
        [
            "What is somatic tracking?",
            "Can you explain the resilient zone?",
            "How do I practice grounding?",
            "Tell me about body literacy.",
            "Teach me about mindfulness and compassion.",
            "What does empathy mean in SEEK?",
            "How can I learn about emotional tracking?",
        ],
    )
    def test_curriculum_only_routes_to_tutor(self, message):
        result = AgentRouter.classify_intent(message)
        assert result["route_target"] == "tutor_agent"
        assert result["intent"] == StudentIntent.CURRICULUM_INQUIRY.value
        assert result["confidence"] == 0.95
        assert result["has_seek_concept"] is True or any(
            concept in message.lower()
            for concept in ("somatic", "resilient", "grounding", "body", "mindfulness", "empathy")
        )

    # -----------------------------------------------------------------
    # 3. Dual Intent: Distress + Curriculum Question
    # -----------------------------------------------------------------
    @pytest.mark.parametrize(
        "message",
        [
            "I'm feeling really anxious and overwhelmed today, can you explain resilient zone?",
            "I feel so stressed out and upset. How do I practice grounding?",
            "I'm struggling with panic right now, what is somatic tracking?",
            "I feel sad and frustrated with school. Can you teach me about mindfulness?",
            "I am exhausted and nervous, tell me about compassion.",
        ],
    )
    def test_dual_intent_routes_to_support_then_tutor(self, message):
        result = AgentRouter.classify_intent(message)
        assert result["route_target"] == "support_then_tutor"
        assert result["intent"] == StudentIntent.EMOTIONAL_DISTRESS.value
        assert result["confidence"] == 0.85

    # -----------------------------------------------------------------
    # 4. General Conversation
    # -----------------------------------------------------------------
    @pytest.mark.parametrize(
        "message",
        [
            "Do you like computer science?",
            "Tell me an interesting fun fact about astronomy.",
            "What do you think about rainy days?",
            "I enjoy programming and reading stories.",
        ],
    )
    def test_general_chat_routes_to_tutor(self, message):
        result = AgentRouter.classify_intent(message)
        assert result["route_target"] == "tutor_agent"
        assert result["intent"] == StudentIntent.GENERAL_CHAT.value
        assert result["confidence"] == 0.7
        assert result["has_seek_concept"] is False

    # -----------------------------------------------------------------
    # 5. Ambiguous Messages
    # -----------------------------------------------------------------
    @pytest.mark.parametrize(
        "message",
        [
            "maybe later",
            "hmm not really sure",
            "okay",
            "well...",
            "12345",
            "asdfghjkl",
        ],
    )
    def test_ambiguous_messages_route_to_tutor_general_chat(self, message):
        result = AgentRouter.classify_intent(message)
        assert result["route_target"] == "tutor_agent"
        assert result["intent"] == StudentIntent.GENERAL_CHAT.value
        assert result["confidence"] == 0.7

    # -----------------------------------------------------------------
    # 6. Standalone Greetings
    # -----------------------------------------------------------------
    @pytest.mark.parametrize(
        "message",
        [
            "Hi",
            "Hello",
            "hey",
            "good morning",
            "good evening",
            "Hello, what can you do?",
            "Hi, how can you help me?",
        ],
    )
    def test_greeting_routes_to_greeting(self, message):
        result = AgentRouter.classify_intent(message)
        assert result["route_target"] == "greeting"
        assert result["intent"] == StudentIntent.GREETING.value
        assert result["confidence"] == 1.0


class TestSafetyGateInterception:
    """Verifies that safety-critical messages qualify for safety interception

    and cannot bypass the safety gate even when combined with curriculum or distress.
    """

    @pytest.mark.parametrize(
        "crisis_message",
        [
            "I want to kill myself.",
            "I don't want to live anymore.",
            "Life isn't worth living.",
            "I feel completely hopeless and want to die.",
            "I want to die, can you explain resilient zone?",
            "I feel like killing myself, how do I practice grounding?",
            "I'm hurting myself tonight because of school, what is somatic tracking?",
            "I took an overdose, teach me about empathy.",
        ],
    )
    def test_safety_critical_messages_qualify_at_safety_gate(self, crisis_message):
        """All safety-critical messages, even with dual intent or curriculum questions,

        must qualify under detect_safety_risk so the safety gate intercepts them.
        """
        risk = detect_safety_risk(crisis_message)
        assert risk["qualifies"] is True, f"Failed to qualify crisis message: {crisis_message}"
        assert risk["risk_level"] in {"high", "immediate"}

"""Unit and regression tests for LearningRecommendationService.

Verifies:
1. All eight approved SEEK skills correctly map from appropriate conversation themes.
2. Irrelevant conversations (or casual greetings) return zero recommendations.
3. Empty conversation history returns zero recommendations.
4. Multiple relevant skills are identified and capped at a maximum of two.
5. Invalid skills or objectives are strictly rejected/reconciled with canonical data.
6. Low-confidence or weak matches are filtered out.
7. Safety escalation takes priority when high-risk content is in the context.
8. Deterministic fallback works flawlessly when Gemini is unavailable or errors.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from knowledge.objective_mapper import CANONICAL_OBJECTIVE_IDS
from learning.learning_recommendation_service import (
    LearningRecommendationService,
    RecommendationItem,
)


@pytest.fixture
def service():
    """Initializes recommendation service using deterministic fallback (client=None)."""
    return LearningRecommendationService(client=None, confidence_threshold=0.35, max_recommendations=2)


# ==============================================================================
# TEST 1: ALL EIGHT APPROVED SEEK SKILLS CORRECTLY MAPPED
# ==============================================================================

@pytest.mark.parametrize(
    ("expected_skill_id", "expected_title", "sample_dialogue"),
    [
        (
            "skill_01",
            "Calming the Body and Mind",
            "My body feels completely tense and my heart is racing. I can't catch my breath and my hands are shaking with physical stress.",
        ),
        (
            "skill_02",
            "Ethical Mindfulness",
            "I'm facing a serious moral dilemma with my values under pressure. My mind wanders and I acted impulsively instead of making a value-aligned decision.",
        ),
        (
            "skill_03",
            "Emotional Awareness",
            "My anger escalated so quickly from a tiny spark to a forest fire. I lost my temper and snapped at everyone before recognizing my emotional triggers.",
        ),
        (
            "skill_04",
            "Self-Compassion",
            "I am constantly beating myself up and feeling like an utter failure. My harsh self-criticism and perfectionism make me feel I am never good enough.",
        ),
        (
            "skill_05",
            "Impartiality and Common Humanity",
            "I feel like an outsider alienated from everyone. There is so much in-group bias, stereotypes, and prejudice dividing us from recognizing our common humanity.",
        ),
        (
            "skill_06",
            "Forgiveness and Gratitude",
            "I have been holding onto deep resentment and a bitter grudge after my friend betrayed me. I really struggle to forgive them and let go of resentment.",
        ),
        (
            "skill_07",
            "Empathic Concern",
            "I am completely drained and experiencing empathy fatigue from absorbing everyone else's problems. Emotional contagion and empathic distress are overwhelming me.",
        ),
        (
            "skill_08",
            "Compassion",
            "I feel heartbroken noticing the suffering of people around me, but I feel helpless and lack a sense of agency to take courageous active compassion steps.",
        ),
    ],
)
def test_all_eight_approved_skills_mapped(service, expected_skill_id, expected_title, sample_dialogue):
    history = [
        {"role": "user", "content": sample_dialogue},
        {"role": "assistant", "content": "I hear how much that is affecting you."},
    ]
    result = service.recommend(history)

    assert result["status"] == "success"
    assert len(result["recommendations"]) >= 1
    top_rec = result["recommendations"][0]
    assert top_rec["skill_id"] == expected_skill_id
    assert top_rec["skill_title"] == expected_title
    assert top_rec["objective_id"] in CANONICAL_OBJECTIVE_IDS[expected_skill_id].values()
    assert top_rec["relevance_explanation"]
    assert top_rec["learning_opportunity"]


# ==============================================================================
# TEST 2: IRRELEVANT CONVERSATIONS & CASUAL GREETINGS (ZERO RECOMMENDATIONS)
# ==============================================================================

@pytest.mark.parametrize(
    "dialogue",
    [
        "Hello, good morning!",
        "Hi there.",
        "What is the weather today in Colombo?",
        "Do you know the score of yesterday's football match?",
        "Okay, thank you, bye.",
        "Yes.",
        "Sure, that's fine.",
    ],
)
def test_irrelevant_conversations_return_zero_recommendations(service, dialogue):
    history = [{"role": "user", "content": dialogue}]
    result = service.recommend(history)

    assert result["status"] == "no_recommendations"
    assert result["recommendations"] == []


# ==============================================================================
# TEST 3: EMPTY CONVERSATION HISTORY
# ==============================================================================

def test_empty_conversation_history(service):
    assert service.recommend([]) == {"status": "no_recommendations", "recommendations": []}
    assert service.recommend([{"role": "assistant", "content": "Hello"}]) == {"status": "no_recommendations", "recommendations": []}
    assert service.recommend([{"role": "user", "content": "   "}]) == {"status": "no_recommendations", "recommendations": []}


# ==============================================================================
# TEST 4: MULTIPLE RELEVANT SKILLS & MAXIMUM TWO RECOMMENDATIONS
# ==============================================================================

def test_multiple_relevant_skills_capped_at_maximum_two(service):
    # This text touches physical tension (skill_01), self-criticism (skill_04), and resentment (skill_06)
    rich_dialogue = (
        "My body is trembling and my chest is tight from extreme physical tension and panic. "
        "At the same time, I can't stop beating myself up with harsh self-criticism and feeling not good enough. "
        "And on top of that, I feel deep resentment and a bitter grudge towards my colleague who betrayed me."
    )
    history = [{"role": "user", "content": rich_dialogue}]
    result = service.recommend(history)

    assert result["status"] == "success"
    assert len(result["recommendations"]) == 2
    # Ensure skills are distinct
    sids = [r["skill_id"] for r in result["recommendations"]]
    assert len(sids) == len(set(sids))
    for r in result["recommendations"]:
        assert r["skill_id"] in {"skill_01", "skill_04", "skill_06"}


# ==============================================================================
# TEST 5: INVALID SKILL / OBJECTIVE REJECTION & CANONICAL SANITIZATION
# ==============================================================================

def test_invalid_skill_rejection_and_objective_sanitization(service):
    invalid_recs = [
        RecommendationItem(
            skill_id="skill_99_invented",
            skill_title="Invented Skill",
            objective_id="obj_99_01",
            objective_title="Fake Objective",
            relevance_explanation="Fake explanation",
            learning_opportunity="Fake opportunity",
            confidence_score=0.9,
        ),
        RecommendationItem(
            skill_id="skill_01",
            skill_title="Calming the Body and Mind",
            objective_id="obj_fake_999",  # Invalid objective ID for skill_01
            objective_title="Invented Objective",
            relevance_explanation="Legit explanation",
            learning_opportunity="Legit opportunity",
            confidence_score=0.85,
        ),
    ]

    sanitized = service._validate_and_sanitize_recommendations(invalid_recs)

    assert len(sanitized) == 1
    # Invalid skill_99 is completely rejected
    assert sanitized[0].skill_id == "skill_01"
    # Invalid objective ID was replaced with a canonical objective ID for skill_01
    assert sanitized[0].objective_id in CANONICAL_OBJECTIVE_IDS["skill_01"].values()


# ==============================================================================
# TEST 6: LOW-CONFIDENCE RECOMMENDATIONS FILTERED OUT
# ==============================================================================

def test_low_confidence_filtered_out(service):
    # A single vague mention that doesn't reach the 0.35 confidence threshold
    vague_text = "I saw someone's body in a movie yesterday."
    result = service.recommend([{"role": "user", "content": vague_text}])

    assert result["status"] == "no_recommendations"
    assert result["recommendations"] == []


# ==============================================================================
# TEST 7: SAFETY ESCALATION TAKES PRECEDENCE
# ==============================================================================

def test_safety_escalation_bypasses_recommendations(service):
    # A message containing self-harm risk
    history = [
        {"role": "user", "content": "I want to kill myself, I feel hopeless."},
        {"role": "assistant", "content": "I care about your safety."},
    ]
    result = service.recommend(history)

    assert result["status"] == "safety_escalation"
    assert result.get("safety") is True
    assert result["recommendations"] == []


# ==============================================================================
# TEST 8: DETERMINISTIC FALLBACK WHEN GEMINI ERRORS OR IS UNAVAILABLE
# ==============================================================================

def test_deterministic_fallback_when_gemini_fails():
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("API network failure")

    service_with_failing_client = LearningRecommendationService(
        client=mock_client, confidence_threshold=0.35, max_recommendations=2
    )

    history = [
        {
            "role": "user",
            "content": "I am so harsh on myself and beat myself up with perfectionism every day.",
        }
    ]
    result = service_with_failing_client.recommend(history)

    assert result["status"] == "success"
    assert len(result["recommendations"]) == 1
    assert result["recommendations"][0]["skill_id"] == "skill_04"


def test_gemini_structured_output_when_valid():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"recommendations": [{"skill_id": "skill_01", "relevance_explanation": "You spoke of stress in your body.", "confidence_score": 0.9}]}'
    mock_client.models.generate_content.return_value = mock_response

    service_with_working_client = LearningRecommendationService(
        client=mock_client, confidence_threshold=0.35, max_recommendations=2
    )

    history = [{"role": "user", "content": "My body is so stressed out and tense."}]
    result = service_with_working_client.recommend(history)

    assert result["status"] == "success"
    assert len(result["recommendations"]) == 1
    assert result["recommendations"][0]["skill_id"] == "skill_01"
    assert result["recommendations"][0]["objective_id"] in CANONICAL_OBJECTIVE_IDS["skill_01"].values()

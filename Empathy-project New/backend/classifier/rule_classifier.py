import re


def classify(block):
    """
    Classifies SEEK Learning PDF blocks into knowledge node types
    and returns a confidence score.

    The classifier is intentionally rule-based because the SEEK PDF
    structure is the authoritative source for the knowledge base.
    """

    text = block.get("text", "").strip()
    text_lower = text.lower()

    font = block.get("font_name", "").lower()

    try:
        font_size = float(block.get("font_size", 0))
    except (ValueError, TypeError):
        font_size = 0

    is_bold = "bold" in font

    # ============================================================
    # 1. EMPTY / INVALID TEXT
    # ============================================================

    if not text:
        return {
            "type": "content",
            "confidence": 0.20
        }

    # Ignore bullet-only blocks such as ● or •
    meaningful_text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)

    if len(meaningful_text.strip()) < 2:
        return {
            "type": "content",
            "confidence": 0.20
        }

    # ============================================================
    # 2. CHAPTER / SKILL DETECTION
    # ============================================================

    # Examples:
    # Skill 1: Calming the Body and Mind
    # Skill 2: Ethical Mindfulness
    # Skill 7: Empathic Concern

    if re.match(r"^skill\s+\d+\s*:", text_lower):
        return {
            "type": "chapter",
            "confidence": 0.98
        }

    # ============================================================
    # 3. OBJECTIVE HEADING DETECTION
    # ============================================================

    # These are headings introducing objectives.
    # They are NOT objectives themselves.

    if text_lower in {
        "module objectives",
        "learning objectives",
        "objectives",
        "module objective"
    }:
        return {
            "type": "objective_heading",
            "confidence": 0.98
        }

    if (
        text_lower.startswith("by the end of this skill")
        or text_lower.startswith("by the end of this module")
        or text_lower.startswith("you will be able to")
    ):
        return {
            "type": "objective_heading",
            "confidence": 0.98
        }

    # ============================================================
    # 4. LEARNING OBJECTIVE DETECTION
    # ============================================================

    # SEEK objectives commonly start with action verbs.

    objective_start = re.compile(
        r"^(define|identify|recognize|describe|explain|apply|"
        r"demonstrate|understand|list|compare|distinguish|use|"
        r"develop|practice|discuss|explore|evaluate|reflect|"
        r"describe|differentiate|summarize|identify|name)\b",
        re.IGNORECASE
    )

    if objective_start.match(text):
        return {
            "type": "learning_objective",
            "confidence": 0.90
        }

    # ============================================================
    # 5. REFLECTION DETECTION
    # ============================================================

    reflection_patterns = [
        r"^reflection$",
        r"^reflections$",
        r"^questions for reflection$",
        r"^reflect on",
        r"^reflect:",
        r"^think about",
        r"^take a moment to reflect"
    ]

    if any(re.search(pattern, text_lower) for pattern in reflection_patterns):
        return {
            "type": "reflection",
            "confidence": 0.92
        }

    # ============================================================
    # 6. ASSESSMENT DETECTION
    # ============================================================

    assessment_patterns = [
        r"^assessment$",
        r"^quiz$",
        r"^test yourself$",
        r"^check your understanding$",
        r"^knowledge check$",
        r"^self[- ]assessment$"
    ]

    if any(re.search(pattern, text_lower) for pattern in assessment_patterns):
        return {
            "type": "assessment",
            "confidence": 0.92
        }

    # ============================================================
    # 7. ACTIVITY DETECTION
    # ============================================================

    # Only classify as activity when the block itself clearly
    # represents an activity, rather than merely mentioning
    # "discussion" somewhere in a paragraph.

    activity_patterns = [
        r"^activity\b",
        r"^activity \d+",
        r"^group activity\b",
        r"^group discussion\b",
        r"^activity prep\b",
        r"^practice activity\b"
    ]

    if any(re.search(pattern, text_lower) for pattern in activity_patterns):
        return {
            "type": "activity",
            "confidence": 0.90
        }

    # ============================================================
    # 8. PRACTICE DETECTION
    # ============================================================

    practice_patterns = [
        r"^practice\b",
        r"^exercise\b",
        r"^grounding\b",
        r"^tracking\b",
        r"^resourcing\b",
        r"^breathing\b",
        r"^meditation\b",
        r"^practice \d+"
    ]

    if any(re.search(pattern, text_lower) for pattern in practice_patterns):
        return {
            "type": "practice",
            "confidence": 0.88
        }

    # ============================================================
    # 9. EXAMPLE DETECTION
    # ============================================================

    example_patterns = [
        r"^imagine\b",
        r"^for example\b",
        r"^consider this example\b",
        r"^here is an example\b"
    ]

    if any(re.search(pattern, text_lower) for pattern in example_patterns):
        return {
            "type": "example",
            "confidence": 0.88
        }

    # ============================================================
    # 10. TOPIC DETECTION
    # ============================================================

    # A topic is normally:
    # - short
    # - bold
    # - heading-like
    # - not a sentence

    words = text.split()

    if (
        is_bold
        and 11 <= font_size <= 18
        and 1 <= len(words) <= 8
        and not text.endswith(".")
        and not text.endswith("?")
        and not text.endswith("!")
    ):
        return {
            "type": "topic",
            "confidence": 0.85
        }

    # ============================================================
    # 11. GENERAL CONTENT
    # ============================================================

    return {
        "type": "content",
        "confidence": 0.70
    }
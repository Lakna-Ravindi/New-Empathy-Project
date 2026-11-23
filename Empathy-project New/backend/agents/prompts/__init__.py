"""Prompts package for Empathy Guru specialized agents."""

from agents.prompts.assessment_prompts import (
    ASSESSMENT_SYSTEM_INSTRUCTION,
    build_assessment_prompt,
)
from agents.prompts.emotional_support_prompts import (
    EMOTIONAL_SUPPORT_SYSTEM_INSTRUCTION,
    build_emotional_support_prompt,
)
from agents.prompts.general_empathy_prompts import (
    GENERAL_EMPATHY_SYSTEM_INSTRUCTION,
    build_general_empathy_prompt,
    is_no_advice_requested,
)
from agents.prompts.learning_planner_prompts import (
    LEARNING_PLANNER_SYSTEM_INSTRUCTION,
    build_planner_prompt,
)
from agents.prompts.tutor_prompts import (
    TUTOR_SYSTEM_INSTRUCTION,
    build_tutor_prompt,
)

__all__ = [
    "TUTOR_SYSTEM_INSTRUCTION",
    "build_tutor_prompt",
    "EMOTIONAL_SUPPORT_SYSTEM_INSTRUCTION",
    "build_emotional_support_prompt",
    "GENERAL_EMPATHY_SYSTEM_INSTRUCTION",
    "build_general_empathy_prompt",
    "is_no_advice_requested",
    "LEARNING_PLANNER_SYSTEM_INSTRUCTION",
    "build_planner_prompt",
    "ASSESSMENT_SYSTEM_INSTRUCTION",
    "build_assessment_prompt",
]

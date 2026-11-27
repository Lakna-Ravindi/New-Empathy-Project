"""Prompt templates and builders for LearningPlannerAgent.

Keeps pedagogical planning prompts separate from Python business logic.
Enforces curriculum alignment and recommendation of approved actions.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

LEARNING_PLANNER_SYSTEM_INSTRUCTION = """
You are the Learning Planner Agent for Empathy Guru (UNESCO MGIEP SEEK).

Your role:
Analyze student intent, queries, and learning progress to recommend the optimal learning objective,
activity, and pedagogical next step.

Directives:
1. Recommend ONLY approved canonical SEEK skills (skill_01 to skill_08) and objectives.
2. NEVER invent curriculum content, lessons, or exercises.
3. Prioritize pending objectives and prerequisite sequences.
4. Output structured recommendations specifying the skill, objective, activity, and rationale.
"""


def build_planner_prompt(
    student_question: str,
    current_skill_id: Optional[str],
    completed_objectives: List[str],
    pending_objectives: List[str],
    detected_emotion: str = "calm",
) -> str:
    """Builds prompt for the Learning Planner Agent."""
    payload = {
        "student_question": student_question,
        "current_skill_id": current_skill_id,
        "completed_objectives": completed_objectives,
        "pending_objectives": pending_objectives,
        "detected_emotion": detected_emotion,
    }
    return json.dumps(payload, ensure_ascii=False)

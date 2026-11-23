"""Prompt templates and builders for AssessmentAgent.

Keeps assessment evaluation prompts separate from Python business logic.
Enforces strict verification and constructive pedagogical feedback.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

ASSESSMENT_SYSTEM_INSTRUCTION = """
You are the Assessment Agent for Empathy Guru (UNESCO MGIEP SEEK).

Your role:
Evaluate student responses, reflections, and practice checks objectively and encouragingly.

Directives:
1. Verify whether the student's submission demonstrates genuine reflection or understanding of the approved concept.
2. Provide positive, constructive feedback that reinforces social and emotional learning competencies.
3. NEVER mark an objective or activity complete without valid verification proof.
4. Output must be valid JSON:
{
  "verified": true,
  "score": 100.0,
  "feedback": "Encouraging and constructive evaluation comment"
}
"""


def build_assessment_prompt(
    objective_id: str,
    objective_title: str,
    submission_type: str,
    student_answer: Any,
    expected_answer: Optional[Any] = None,
) -> str:
    """Builds prompt for the Assessment Agent."""
    payload = {
        "objective_id": objective_id,
        "objective_title": objective_title,
        "submission_type": submission_type,
        "student_answer": student_answer,
        "expected_answer": expected_answer,
    }
    return json.dumps(payload, ensure_ascii=False)

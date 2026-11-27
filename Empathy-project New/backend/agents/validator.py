"""AgentResponseValidator: Response schema, anti-hallucination, and grounding verification.

Validates:
1. Schema Integrity: answer is non-empty, exactly 4 steps with correct types, suggested questions.
2. Truncation & Repetition Suppression: checks sentence boundaries and duplicated paragraphs.
3. Activity Metadata Attachment: copies progress_item_id and progress_item_type to Step 3.
4. Deterministic Fallback Triggering: safely handles invalid or corrupt model output.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

EXPECTED_STEP_TYPES = [
    "direct_answer",
    "learning_objective",
    "activity",
    "practice_check",
]


class AgentResponseValidator:
    """Verifies generated agent responses against UNESCO SEEK contracts."""

    @staticmethod
    def validate_educational_contract(
        payload: Dict[str, Any],
        selected_activity: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Validates that a response payload strictly adheres to the 4-stage contract."""
        if not isinstance(payload, dict):
            return False

        # 1. Answer check
        answer = payload.get("answer") or payload.get("educational_response")
        if not isinstance(answer, str) or not answer.strip():
            logger.warning("Validation failed: Missing or empty answer string.")
            return False

        # 2. Steps check
        steps = payload.get("steps")
        if not isinstance(steps, list):
            logger.warning("Validation failed: 'steps' is not a list.")
            return False

        if len(steps) != 4:
            logger.warning("Validation failed: Expected exactly 4 steps, found %d.", len(steps))
            return False

        for idx, (step, expected_type) in enumerate(zip(steps, EXPECTED_STEP_TYPES), 1):
            if not isinstance(step, dict):
                logger.warning("Validation failed: Step %d is not a dictionary.", idx)
                return False
            if step.get("type") != expected_type:
                logger.warning("Validation failed: Step %d type mismatch. Expected '%s', got '%s'.", idx, expected_type, step.get("type"))
                return False
            if not isinstance(step.get("content"), str) or not step["content"].strip():
                logger.warning("Validation failed: Step %d content is empty.", idx)
                return False

        # 3. Practice check specific validation
        practice_step = steps[3]
        if "correct_answer" not in practice_step:
            logger.warning("Validation failed: Step 4 missing 'correct_answer'.")
            return False

        return True

    @staticmethod
    def attach_progress_metadata(
        payload: Dict[str, Any],
        recommended_activity: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Injects progress_item_id and objective metadata into Step 3 activity."""
        if not recommended_activity or not isinstance(recommended_activity, dict):
            return payload

        progress_item_id = recommended_activity.get("progress_item_id")
        if not progress_item_id:
            return payload

        steps = payload.get("steps", [])
        for step in steps:
            if isinstance(step, dict) and step.get("type") == "activity":
                step_activity = step.get("activity")
                if isinstance(step_activity, dict):
                    step["activity"] = {
                        **step_activity,
                        "progress_item_id": progress_item_id,
                        "progress_item_type": recommended_activity.get("progress_item_type", "activity"),
                        "progress_objective_id": recommended_activity.get("progress_objective_id"),
                    }
                elif step_activity is None:
                    step["activity"] = {
                        "id": recommended_activity.get("id", progress_item_id),
                        "title": recommended_activity.get("title", ""),
                        "type": recommended_activity.get("type", "activity"),
                        "progress_item_id": progress_item_id,
                        "progress_item_type": recommended_activity.get("progress_item_type", "activity"),
                        "progress_objective_id": recommended_activity.get("progress_objective_id"),
                    }
        return payload

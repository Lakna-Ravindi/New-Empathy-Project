import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from learning.gemini_responder import (
    _fallback_response,
    _parse_response,
    generate_educational_response,
)


def response_payload(activity):
    return json.dumps({
        "answer": "A short answer.",
        "steps": [
            {"step": 1, "type": "direct_answer", "content": "Answer."},
            {"step": 2, "type": "learning_objective", "content": "Objective."},
            {
                "step": 3,
                "type": "activity",
                "content": "Try this.",
                "activity": activity,
            },
            {
                "step": 4,
                "type": "practice_check",
                "content": "Check.",
                "question_type": "true_false",
                "question": "Is this useful?",
            },
        ],
    })


class GeminiResponderTests(unittest.TestCase):
    def test_long_fallback_is_more_detailed_and_structured(self):
        context = {
            "student_question": "How can I calm down?",
            "skill": {"title": "Regulation"},
            "learning_objective": {"title": "Notice emotions"},
            "recommended_activity": {
                "id": "activity_01",
                "title": "Grounding",
                "type": "activity",
            },
        }

        short = _fallback_response(context, "short")
        long = _fallback_response(context, "long")

        self.assertGreater(len(long["answer"]), len(short["answer"]))
        self.assertIn("1)", long["answer"])
        self.assertIn("2)", long["answer"])
        self.assertEqual(long["steps"][0]["content"], long["answer"])
        self.assertIn("Learning objective", long["steps"][1]["content"])

    def test_activity_metadata_must_match_selected_context(self):
        selected = {
            "id": "activity_01",
            "title": "Grounding",
            "type": "activity",
        }
        parsed = _parse_response(response_payload(selected), selected)
        self.assertEqual(parsed["steps"][2]["activity"], selected)

        with self.assertRaises(ValueError):
            _parse_response(
                response_payload({
                    "id": "invented",
                    "title": "Unapproved activity",
                    "type": "activity",
                }),
                selected,
            )

    def test_missing_activity_is_represented_as_unavailable(self):
        context = {
            "student_question": "What is empathy?",
            "skill": {"title": "Self compassion"},
            "learning_objective": {"title": "Understand empathy"},
            "recommended_activity": None,
        }
        parsed = _parse_response(
            response_payload(None),
            None,
        )
        self.assertIsNone(parsed["steps"][2]["activity"])
        fallback = _fallback_response(context)
        self.assertIsNone(fallback["steps"][2]["activity"])
        self.assertNotIn("activity_unassigned", json.dumps(fallback))

    def test_null_activity_metadata_is_rejected_when_activity_was_selected(self):
        selected = {"id": "activity_01", "title": "Grounding", "type": "activity"}
        with self.assertRaises(ValueError):
            _parse_response(response_payload(None), selected)

    @patch("learning.gemini_responder.genai.Client")
    def test_exam_question_prompt_includes_relevant_seek_content(self, client_class):
        selected = {"id": "activity_01", "title": "Grounding", "type": "activity"}
        client_class.return_value.models.generate_content.return_value = Mock(
            text=response_payload(selected),
            candidates=[],
        )
        context = {
            "status": "learning_path_selected",
            "student_question": "How can I stay calm during exams?",
            "skill": {"id": "skill_01", "title": "Calming the body"},
            "topic": {},
            "learning_objective": {"title": "Stay regulated"},
            "recommended_activity": selected,
            "question_relevant_content": [{
                "id": "node_0032",
                "title": "Resilient zone",
                "content": "Staying calm during a tough exam can reflect the resilient zone.",
                "page": 6,
            }],
            "matched_skills": [],
            "matched_keywords": ["calm"],
            "source_page": 6,
        }
        result = generate_educational_response(context)
        prompt = client_class.return_value.models.generate_content.call_args.kwargs[
            "contents"
        ]
        self.assertIn("How can I stay calm during exams?", prompt)
        self.assertIn("Staying calm during a tough exam", prompt)
        self.assertEqual(result["answer"], "A short answer.")

    @patch("learning.gemini_responder.time.sleep")
    @patch("learning.gemini_responder.genai.Client")
    def test_truncated_json_after_retries_returns_controlled_fallback(
        self, client_class, _sleep
    ):
        response = Mock(text='{"answer":"truncated"', candidates=[])
        client_class.return_value.models.generate_content.return_value = response
        context = {
            "status": "learning_path_selected",
            "student_question": "What is empathy?",
            "skill": {"title": "Self compassion"},
            "topic": {},
            "learning_objective": None,
            "recommended_activity": None,
            "question_relevant_content": [],
            "matched_skills": [],
            "matched_keywords": [],
            "source_page": None,
        }
        result = generate_educational_response(context)
        self.assertEqual(len(result["steps"]), 4)
        self.assertIsNone(result["steps"][2]["activity"])
        self.assertEqual(client_class.return_value.models.generate_content.call_count, 3)


if __name__ == "__main__":
    unittest.main()

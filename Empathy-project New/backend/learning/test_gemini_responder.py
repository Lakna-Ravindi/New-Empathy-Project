import json
import os
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
import learning.gemini_responder as gemini_responder


def response_payload(activity):
    return json.dumps({
        "answer": "A short answer.",
        "suggested_questions": [
            "What does this idea mean?",
            "How does it relate to the objective?",
            "Can you explain this concept?",
        ],
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
    def test_fallback_is_concise_and_structured_for_both_answer_styles(self):
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

        self.assertEqual(long["answer"], short["answer"])
        self.assertNotIn("The approved SEEK content says", long["answer"])
        self.assertEqual(long["steps"][0]["content"], long["answer"])
        self.assertIn("Notice emotions", long["steps"][1]["content"])
        self.assertIn("instructions are not available", long["steps"][2]["content"])

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
        self.assertEqual(len(fallback["suggested_questions"]), 3)
        self.assertTrue(all(
            "Understand empathy" in question
            for question in fallback["suggested_questions"]
        ))

    def test_suggested_questions_must_be_three_or_four_non_empty_strings(self):
        for suggestions in (
            None,
            [],
            ["one", "two"],
            ["one", "", "three"],
            ["one", "two", 3],
            ["one", "two", "three", "four", "five"],
        ):
            payload = json.loads(response_payload(None))
            payload["suggested_questions"] = suggestions
            with self.subTest(suggestions=suggestions), self.assertRaises(ValueError):
                _parse_response(json.dumps(payload), None)

        payload = json.loads(response_payload(None))
        payload["suggested_questions"] = [
            "  What is this?  ",
            "How does it work?",
            "Why does it matter?",
            "Can you give an example?",
        ]
        parsed = _parse_response(json.dumps(payload), None)
        self.assertEqual(len(parsed["suggested_questions"]), 4)
        self.assertEqual(parsed["suggested_questions"][0], "What is this?")

    def test_fallback_does_not_invent_objective_activity_or_seek_instructions(self):
        fallback = _fallback_response({
            "student_question": "What should I do?",
            "skill": {},
            "learning_objective": None,
            "recommended_activity": None,
            "question_relevant_content": [],
        })

        self.assertIn("don't have enough approved information", fallback["answer"])
        self.assertIn("No approved learning objective", fallback["steps"][1]["content"])
        self.assertIsNone(fallback["steps"][2]["activity"])
        self.assertEqual(fallback["suggested_questions"], [])

    def test_fallback_prioritizes_urgent_physical_symptoms(self):
        fallback = _fallback_response({
            "student_question": "I have severe chest tightness and difficulty breathing.",
            "learning_objective": None,
            "recommended_activity": None,
            "question_relevant_content": [],
        })

        self.assertIn("seek immediate medical help now", fallback["answer"])
        self.assertNotIn("caused by stress", fallback["answer"])

    def test_fallback_questions_use_next_objective_approved_content(self):
        fallback = _fallback_response({
            "status": "learning_path_selected",
            "student_question": "Continue learning.",
            "learning_objective": {"title": "Completed objective"},
            "suggestion_learning_objective": {
                "title": "Practice self-compassion",
            },
            "question_relevant_content": [{
                "title": "Completed topic",
                "content": "Previously learned material.",
            }],
            "suggestion_relevant_content": [{
                "title": "Common humanity",
                "content": "People can recognize that suffering is part of the shared human experience.",
            }, {
                "title": "Self-kindness",
                "content": "Treating oneself with care during difficult moments.",
            }],
            "recommended_activity": None,
        })

        self.assertEqual(len(fallback["suggested_questions"]), 4)
        self.assertTrue(all(
            any(topic in question for topic in ("Common humanity", "Self-kindness"))
            for question in fallback["suggested_questions"]
        ))
        self.assertTrue(all(
            "Completed topic" not in question
            for question in fallback["suggested_questions"]
        ))

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    @patch("learning.gemini_responder.genai.Client")
    def test_missing_relevant_seek_content_skips_gemini(self, client_class):
        result = generate_educational_response({
            "status": "learning_path_selected",
            "student_question": "Unrelated question",
            "skill": {},
            "learning_objective": None,
            "recommended_activity": None,
            "question_relevant_content": [],
        })

        client_class.assert_not_called()
        self.assertEqual(result["suggested_questions"], [])
        self.assertEqual(len(result["steps"]), 4)

    def test_null_activity_metadata_is_rejected_when_activity_was_selected(self):
        selected = {"id": "activity_01", "title": "Grounding", "type": "activity"}
        with self.assertRaises(ValueError):
            _parse_response(response_payload(None), selected)

    def test_question_type_aliases_are_normalized(self):
        payload = json.loads(response_payload(None))
        payload["steps"][3]["question_type"] = "YesNo"
        parsed = _parse_response(json.dumps(payload), None)
        self.assertEqual(parsed["steps"][3]["question_type"], "yes_no")

        payload = json.loads(response_payload(None))
        payload["steps"][3]["question_type"] = "TrueFalse"
        parsed = _parse_response(json.dumps(payload), None)
        self.assertEqual(parsed["steps"][3]["question_type"], "true_false")

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
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
        self.assertIn("suggestion_learning_objective", prompt)
        self.assertEqual(len(result["suggested_questions"]), 3)
        self.assertEqual(result["answer"], "A short answer.")

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    @patch("learning.gemini_responder.genai.Client")
    def test_rate_limit_fallback_generates_grounded_suggestions(self, client_class):
        class RateLimitError(Exception):
            code = 429

        client_class.side_effect = RateLimitError("rate limited")
        context = {
            "status": "learning_path_selected",
            "student_question": "What is empathy?",
            "skill": {"title": "Self compassion"},
            "learning_objective": {"title": "Understand empathy"},
            "recommended_activity": None,
            "question_relevant_content": [{
                "title": "Empathy",
                "content": "Empathy means understanding another person's feelings.",
            }],
        }
        with patch.object(gemini_responder, "ClientError", RateLimitError):
            result = generate_educational_response(context)

        self.assertIsInstance(result["answer"], str)
        self.assertEqual(len(result["steps"]), 4)
        self.assertEqual(len(result["suggested_questions"]), 3)
        self.assertEqual(result["steps"][2]["activity"], None)

    @patch("learning.gemini_responder.time.sleep")
    @patch("learning.gemini_responder.genai.Client")
    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    def test_invalid_gemini_suggestions_use_grounded_fallback(
        self, client_class, _sleep
    ):
        payload = json.loads(response_payload(None))
        payload["suggested_questions"] = []
        client_class.return_value.models.generate_content.return_value = Mock(
            text=json.dumps(payload),
            candidates=[],
        )
        context = {
            "status": "learning_path_selected",
            "student_question": "Continue learning.",
            "skill": {"id": "skill_01", "title": "Self-compassion"},
            "learning_objective": {"title": "Completed objective"},
            "suggestion_learning_objective": {
                "title": "Practice self-compassion",
            },
            "recommended_activity": None,
            "question_relevant_content": [{
                "title": "Completed content",
                "content": "Previously learned material.",
            }],
            "suggestion_relevant_content": [{
                "title": "Common humanity",
                "content": "People share difficult experiences.",
            }],
        }

        result = generate_educational_response(context)

        self.assertEqual(
            client_class.return_value.models.generate_content.call_count,
            3,
        )
        self.assertEqual(len(result["suggested_questions"]), 3)
        self.assertTrue(all(
            "Common humanity" in question
            for question in result["suggested_questions"]
        ))

    @patch("learning.gemini_responder.time.sleep")
    @patch("learning.gemini_responder.genai.Client")
    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    def test_server_failure_retries_then_returns_consistent_fallback(
        self, client_class, sleep
    ):
        class ServiceUnavailable(Exception):
            pass

        client_class.return_value.models.generate_content.side_effect = (
            ServiceUnavailable("unavailable")
        )
        context = {
            "status": "learning_path_selected",
            "student_question": "What is empathy?",
            "skill": {"title": "Self compassion"},
            "topic": {},
            "learning_objective": {"title": "Understand empathy"},
            "recommended_activity": None,
            "question_relevant_content": [{
                "title": "Empathy",
                "content": "Empathy means understanding another person's feelings.",
            }],
            "source_page": 1,
        }
        with patch.object(gemini_responder, "ServerError", ServiceUnavailable):
            result = generate_educational_response(context)

        self.assertEqual(
            client_class.return_value.models.generate_content.call_count,
            3,
        )
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(len(result["suggested_questions"]), 3)
        self.assertEqual(len(result["steps"]), 4)

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
            "question_relevant_content": [{
                "title": "Empathy",
                "content": "Approved SEEK content about empathy.",
            }],
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

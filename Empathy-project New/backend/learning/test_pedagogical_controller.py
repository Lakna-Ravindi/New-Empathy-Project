import json
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from learning.pedagogical_controller import (
    answer_student_question,
    belongs_to_skill,
    expand_terms,
    find_skill_mapping,
    get_terms,
    process_student_question,
    QUESTION_RELEVANT_CONTENT_LIMIT,
    score_text,
)


def load_fixture(name):
    with (ROOT / "output" / name).open(encoding="utf-8") as file:
        return json.load(file)


class PedagogicalControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.knowledge_base = load_fixture("knowledge_base.json")
        cls.keyword_map = load_fixture("keyword_skill_map.json")

    def test_punctuation_is_normalized_and_generic_words_are_filtered(self):
        context = process_student_question(
            "I feel overwhelmed before exams. How can I calm down?",
            self.knowledge_base,
            self.keyword_map,
        )

        self.assertIn("exam", context["detected_terms"])
        self.assertNotIn("down", context["detected_terms"])
        self.assertEqual(
            len(context["matched_keywords"]),
            len(set(context["matched_keywords"])),
        )
        self.assertTrue(all("," not in keyword for keyword in context["matched_keywords"]))
        self.assertFalse(
            any(len(keyword.split()) > 4 for keyword in context["matched_keywords"])
        )
        self.assertNotIn(
            "attention back to the present moment creating a sense of stability and calm",
            context["matched_keywords"],
        )

    def test_common_word_forms_are_normalized_without_broad_stemming(self):
        self.assertEqual(get_terms("exams"), {"exam"})
        self.assertEqual(get_terms("exam"), {"exam"})
        self.assertEqual(
            expand_terms("How can I stay calm during exams?"),
            {"exam", "stay", "calm", "during"},
        )
        self.assertEqual(
            expand_terms("I feel stressed before an exam"),
            {"stress", "exam"},
        )

    def test_existing_emotion_matching_is_preserved(self):
        for emotion in ("overwhelmed", "anxious", "angry"):
            with self.subTest(emotion=emotion):
                context = process_student_question(
                    f"I feel {emotion}.",
                    self.knowledge_base,
                    self.keyword_map,
                )
                self.assertNotEqual(context["status"], "no_match")

    def test_stay_calm_during_exams_selects_approved_skill(self):
        context = process_student_question(
            "How can I stay calm during exams?",
            self.knowledge_base,
            self.keyword_map,
        )

        self.assertEqual(context["status"], "learning_path_selected")
        self.assertTrue(context["matched_skills"])
        self.assertTrue(
            any(
                mapping.get("source_node_id") in {
                    item["id"] for item in self.knowledge_base
                }
                for recommendation in context["matched_skills"]
                for mapping in recommendation["mappings"]
            )
        )

    def test_actionable_exam_question_does_not_require_repeated_confirmation(self):
        context = process_student_question(
            "How can I stay calm during exams?",
            self.knowledge_base,
            self.keyword_map,
        )
        self.assertFalse(context["learning_confirmation_required"])
        self.assertEqual(context["recommended_activity"]["id"], "node_0058")

    def test_empathy_question_does_not_reuse_exam_content(self):
        context = process_student_question(
            "What is empathy?",
            self.knowledge_base,
            self.keyword_map,
        )
        self.assertNotEqual(context["skill"]["id"], "node_0001")
        self.assertTrue(context["question_relevant_content"])
        self.assertFalse(
            any(
                "exam" in (
                    f"{chunk.get('title', '')} {chunk.get('content', '')}"
                ).lower()
                for chunk in context["question_relevant_content"]
            )
        )

    def test_question_relevant_content_is_scoped_ranked_and_bounded(self):
        context = process_student_question(
            "How can I stay calm during exams?",
            self.knowledge_base,
            self.keyword_map,
        )

        chunks = context["question_relevant_content"]
        self.assertTrue(chunks)
        self.assertLessEqual(len(chunks), QUESTION_RELEVANT_CONTENT_LIMIT)
        self.assertIn("node_0032", {chunk["id"] for chunk in chunks})
        self.assertTrue(all(chunk["type"] == "content" for chunk in chunks))
        self.assertTrue(all(chunk["content"] for chunk in chunks))
        self.assertTrue(all(chunk["page"] is not None for chunk in chunks))
        self.assertEqual(
            chunks,
            sorted(chunks, key=lambda chunk: (-(
                score_text(
                    f"{chunk['title']} {chunk['content']}",
                    expand_terms("How can I stay calm during exams?"),
                )
            ), chunk["id"])),
        )

        nodes_by_id = {
            node["id"]: node
            for node in self.knowledge_base
            if node.get("id")
        }
        self.assertTrue(
            all(
                belongs_to_skill(nodes_by_id[chunk["id"]], "node_0001", nodes_by_id)
                for chunk in chunks
            )
        )

    def test_no_relevant_content_returns_empty_list(self):
        knowledge_base = [
            {
                "id": "skill-a",
                "type": "chapter",
                "title": "Skill A",
                "content": "Skill A",
                "parent_id": None,
            },
            {
                "id": "objective-a",
                "type": "learning_objective",
                "title": "Understand empathy",
                "content": "Understand empathy",
                "parent_id": "skill-a",
            },
            {
                "id": "content-a",
                "type": "content",
                "title": "A different topic",
                "content": "This content is about gardening.",
                "page": 2,
                "parent_id": "skill-a",
            },
        ]
        context = process_student_question(
            "I want to learn about empathy.",
            knowledge_base,
            [{
                "keyword": "empathy",
                "mapping_status": "mapped",
                "source_node_id": "skill-a",
                "source_node_type": "chapter",
                "skill_id": "skill-a",
                "skill": "Skill A",
            }],
        )
        self.assertEqual(context["question_relevant_content"], [])

    def test_multiple_skills_for_one_keyword_are_retained(self):
        mappings = [
            {
                "keyword": "grounding",
                "mapping_status": "mapped",
                "source_node_id": "skill-a",
                "source_node_type": "chapter",
                "skill_id": "skill-a",
                "skill": "Skill A",
            },
            {
                "keyword": "grounding,",
                "mapping_status": "mapped",
                "source_node_id": "skill-b",
                "source_node_type": "chapter",
                "skill_id": "skill-b",
                "skill": "Skill B",
            },
        ]
        matches, _, _ = find_skill_mapping("I need grounding.", mappings)
        self.assertEqual({match["skill_id"] for match in matches}, {"skill-a", "skill-b"})
        self.assertEqual({match["keyword"] for match in matches}, {"grounding"})

    def test_invalid_source_nodes_are_ignored(self):
        context = process_student_question(
            "I need grounding.",
            self.knowledge_base,
            [{
                "keyword": "grounding",
                "mapping_status": "mapped",
                "source_node_id": "missing-node",
                "source_node_type": "chapter",
                "skill_id": "missing-skill",
                "skill": "Unapproved skill",
            }],
        )
        self.assertEqual(context["status"], "no_match")

    def test_no_relevant_activity_is_explicit(self):
        knowledge_base = [
            {
                "id": "skill-a",
                "type": "chapter",
                "title": "Skill A",
                "content": "Skill A",
                "parent_id": None,
            },
            {
                "id": "objective-a",
                "type": "learning_objective",
                "title": "Understand empathy",
                "content": "Understand empathy",
                "parent_id": "skill-a",
            },
        ]
        context = process_student_question(
            "I want to learn about empathy.",
            knowledge_base,
            [{
                "keyword": "empathy",
                "mapping_status": "mapped",
                "source_node_id": "skill-a",
                "source_node_type": "chapter",
                "skill_id": "skill-a",
                "skill": "Skill A",
            }],
        )
        self.assertIsNone(context["recommended_activity"])
        self.assertEqual(
            context["recommendations"][0]["activity_selection_reason"],
            "no_relevant_activity_for_selected_skill_and_question",
        )

    def test_empathy_and_compassion_retain_relevant_skills(self):
        context = process_student_question(
            "I want to learn about empathy and compassion.",
            self.knowledge_base,
            self.keyword_map,
        )
        skill_ids = {item["skill"]["id"] for item in context["matched_skills"]}
        self.assertIn("node_0187", skill_ids)
        self.assertIn("node_0416", skill_ids)
        self.assertIn("compassion", context["detected_keywords"])

    def test_direct_keywords_are_separate_from_alias_expansion(self):
        context = process_student_question(
            "I feel overwhelmed and anxious before exams. How can I calm down?",
            self.knowledge_base,
            self.keyword_map,
        )
        self.assertEqual(
            set(context["detected_keywords"]),
            {"overwhelmed", "anxious", "calm"},
        )
        self.assertNotIn("grounding", context["detected_keywords"])
        self.assertIn("grounding", context["matched_keywords"])

    def test_primary_recommendation_has_valid_activity_when_one_exists(self):
        context = process_student_question(
            "I feel overwhelmed and anxious before exams. How can I calm down?",
            self.knowledge_base,
            self.keyword_map,
        )
        self.assertEqual(context["skill"]["id"], "node_0001")
        self.assertEqual(context["recommended_activity"]["id"], "node_0058")
        self.assertEqual(
            context["recommendations"][0]["activity_selection_reason"],
            "selected_relevant_activity",
        )

    def test_missing_activity_is_safe_and_explicit(self):
        context = process_student_question(
            "I want to learn about empathy and compassion.",
            self.knowledge_base,
            self.keyword_map,
        )
        for recommendation in context["recommendations"]:
            if recommendation["recommended_activity"] is None:
                self.assertEqual(
                    recommendation["activity_selection_reason"],
                    "no_relevant_activity_for_selected_skill_and_question",
                )

    def test_compassion_uses_valid_reflection_when_ranked_primary(self):
        context = process_student_question(
            "I want to learn about empathy and compassion.",
            self.knowledge_base,
            self.keyword_map,
        )
        self.assertEqual(context["skill"]["id"], "node_0416")
        self.assertEqual(context["recommended_activity"]["id"], "node_0452")

    @patch("learning.pedagogical_controller.generate_educational_response")
    def test_answer_style_is_preserved_for_short_mode(self, responder):
        responder.return_value = {"answer": "Brief.", "steps": []}

        result = answer_student_question(
            "I feel overwhelmed before exams.",
            self.knowledge_base,
            self.keyword_map,
            "short",
        )

        self.assertEqual(result["learning_context"]["answer_style"], "short")
        responder.assert_called_once()
        self.assertEqual(responder.call_args.args[1], "short")

    @patch("learning.pedagogical_controller.generate_educational_response")
    def test_answer_style_is_preserved_for_long_mode(self, responder):
        responder.return_value = {"answer": "Detailed.", "steps": ["step"]}

        result = answer_student_question(
            "I feel overwhelmed before exams.",
            self.knowledge_base,
            self.keyword_map,
            "long",
        )

        self.assertEqual(result["learning_context"]["answer_style"], "long")
        responder.assert_called_once()
        self.assertEqual(responder.call_args.args[1], "long")

    def test_invalid_answer_style_is_rejected(self):
        with self.assertRaises(ValueError):
            answer_student_question(
                "I feel overwhelmed before exams.",
                self.knowledge_base,
                self.keyword_map,
                "medium",
            )

    @patch("learning.pedagogical_controller.generate_educational_response")
    def test_answer_style_defaults_to_short(self, responder):
        responder.return_value = {"answer": "Brief.", "steps": []}

        result = answer_student_question(
            "I feel overwhelmed before exams.",
            self.knowledge_base,
            self.keyword_map,
        )

        self.assertEqual(result["learning_context"]["answer_style"], "short")
        self.assertEqual(responder.call_args.args[1], "short")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "backend"))
    unittest.main()

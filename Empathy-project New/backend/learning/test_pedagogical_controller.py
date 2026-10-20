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
    rank_learning_activities,
    score_text,
)
from learning.interaction_store import LearningStore


def load_fixture(name):
    with (ROOT / "output" / name).open(encoding="utf-8") as file:
        return json.load(file)


class PedagogicalControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.knowledge_base = load_fixture("knowledge_base.json")
        cls.keyword_map = load_fixture("keyword_skill_map.json")
        cls.progress_structure = load_fixture("progress_structure.json")

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

    def test_runtime_retrieval_uses_content_and_highlights_without_map(self):
        highlighted_keywords = load_fixture("highlighted_keywords.json")
        context = process_student_question(
            "How can I stay calm during exams?",
            self.knowledge_base,
            highlighted_keywords,
        )

        self.assertEqual(context["status"], "learning_path_selected")
        self.assertEqual(context["skill"]["id"], "node_0001")
        self.assertTrue(context["learning_objective"]["id"])
        self.assertTrue(context["question_relevant_content"])

    def test_runtime_retrieval_does_not_require_keyword_map_records(self):
        context = process_student_question(
            "What is empathy?",
            self.knowledge_base,
            [],
        )

        self.assertEqual(context["status"], "learning_path_selected")
        self.assertEqual(context["skill"]["id"], "node_0147")

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
            self.progress_structure,
        )
        self.assertFalse(context["learning_confirmation_required"])
        self.assertEqual(context["recommended_activity"]["id"], "reflection_01_03")

    def test_objective_without_authored_activity_does_not_recommend_another_objectives_activity(self):
        nodes = [
            {
                "id": "skill-a",
                "type": "chapter",
                "title": "Skill A",
                "content": "",
                "parent_id": None,
            },
            {
                "id": "objective-a",
                "type": "learning_objective",
                "title": "Objective A",
                "content": "",
                "parent_id": "skill-a",
                "objective_id": "objective-a",
                "skill_id": "skill-a",
            },
            {
                "id": "activity-b",
                "type": "activity",
                "title": "Activity for another objective",
                "content": "An unrelated activity",
                "parent_id": "skill-a",
            },
        ]
        nodes_by_id = {node["id"]: node for node in nodes}
        structure = {
            "chapters": [{
                "chapter_id": "skill-a",
                "objectives": [{
                    "objective_id": "objective-a",
                    "items": [],
                }],
            }],
        }

        activities = rank_learning_activities(
            "skill-a",
            nodes[1],
            nodes,
            nodes_by_id,
            {"unrelated"},
            structure,
        )

        self.assertEqual(activities, [])

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
        self.assertEqual(chunks[0]["id"], "node_0027")
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

    @patch("learning.pedagogical_controller.generate_educational_response")
    def test_no_match_returns_empty_suggestions_without_calling_gemini(self, responder):
        result = answer_student_question(
            "Tell me about unrelated engine maintenance.",
            [],
            [],
        )

        self.assertEqual(result["status"], "no_match")
        self.assertEqual(result["suggested_questions"], [])
        responder.assert_not_called()

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
        self.assertIn("node_0001", skill_ids)
        self.assertIn("node_0147", skill_ids)
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
            self.progress_structure,
        )
        self.assertEqual(context["skill"]["id"], "node_0001")
        self.assertEqual(context["recommended_activity"]["id"], "reflection_01_02")
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
            self.progress_structure,
        )
        self.assertEqual(context["skill"]["id"], "node_0147")
        self.assertTrue(
            context["recommended_activity"]["id"].startswith("activity_04_")
        )

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
        self.assertEqual(result["suggested_questions"], [])

    @patch("learning.pedagogical_controller.generate_educational_response")
    def test_completed_objective_uses_next_incomplete_seek_content_for_suggestions(
        self, responder
    ):
        question = "How can I stay calm during exams?"
        selected_context = process_student_question(
            question,
            self.knowledge_base,
            self.keyword_map,
            self.progress_structure,
        )
        current_objective_id = selected_context["learning_objective"]["objective_id"]
        skill_id = selected_context["skill"]["id"]
        chapter_id = selected_context["learning_objective"]["skill_id"]
        next_content_node = next(
            node
            for node in self.knowledge_base
            if node.get("type") == "content"
            and node.get("skill_id") == chapter_id
            and node.get("objective_id")
            and node.get("objective_id") != current_objective_id
        )
        next_objective_id = next_content_node["objective_id"]
        responder.return_value = {
            "answer": "Grounded answer.",
            "steps": [],
            "suggested_questions": ["Question one?", "Question two?", "Question three?"],
        }
        structure = {
            "chapters": [{
                "chapter_id": chapter_id,
                "objectives": [
                    {
                        "objective_id": current_objective_id,
                        "title": "Current objective",
                        "content": "Current objective content",
                    },
                    {
                        "objective_id": next_objective_id,
                        "title": "Next objective",
                        "content": "Next objective content",
                    },
                ],
            }],
        }

        result = answer_student_question(
            question,
            self.knowledge_base,
            self.keyword_map,
            progress_documents=[{
                "skill_id": chapter_id,
                "completed_objective_ids": [current_objective_id],
            }],
            progress_structure=structure,
        )

        responder_context = responder.call_args.args[0]
        self.assertEqual(
            responder_context["suggestion_learning_objective"]["objective_id"],
            next_objective_id,
        )
        self.assertTrue(responder_context["suggestion_relevant_content"])
        self.assertTrue(all(
            item["objective_id"] == next_objective_id
            for item in responder_context["suggestion_relevant_content"]
        ))
        self.assertEqual(result["suggested_questions"], [
            "Question one?", "Question two?", "Question three?"
        ])

    @patch("learning.pedagogical_controller.generate_educational_response")
    def test_saved_completed_objectives_are_skipped_when_selecting_guidance(
        self, responder
    ):
        question = "Continue learning."
        current_objective_id = "objective_current"
        completed_next_id = "objective_already_complete"
        expected_objective_id = "objective_next"
        selected_context = {
            "student_question": question,
            "status": "learning_path_selected",
            "skill": {"id": "skill_test", "title": "Approved skill"},
            "learning_objective": {
                "id": current_objective_id,
                "objective_id": current_objective_id,
                "skill_id": "skill_test",
                "title": "Current objective",
            },
            "question_relevant_content": [],
        }
        knowledge_base = [
            {
                "id": f"content_{objective_id}",
                "type": "content",
                "skill_id": "skill_test",
                "objective_id": objective_id,
                "title": f"Approved content for {objective_id}",
                "content": f"Approved SEEK material about {objective_id}.",
            }
            for objective_id in (
                current_objective_id,
                completed_next_id,
                expected_objective_id,
            )
        ]
        structure = {
            "chapters": [{
                "chapter_id": "skill_test",
                "title": "Approved skill",
                "objectives": [
                    {
                        "objective_id": objective_id,
                        "title": f"Objective {objective_id}",
                        "content": f"Objective content for {objective_id}.",
                    }
                    for objective_id in (
                        current_objective_id,
                        completed_next_id,
                        expected_objective_id,
                    )
                ],
            }],
        }
        responder.return_value = {
            "answer": "Continue with the next incomplete objective.",
            "steps": [],
            "suggested_questions": [],
        }

        with patch(
            "learning.pedagogical_controller.process_student_question",
            return_value=selected_context,
        ):
            result = answer_student_question(
                question,
                knowledge_base,
                progress_documents=[{
                    "skill_id": "skill_test",
                    "completed_objective_ids": [
                        current_objective_id,
                        completed_next_id,
                    ],
                }],
                progress_structure=structure,
            )

        self.assertEqual(
            result["learning_context"]["learning_objective"]["objective_id"],
            expected_objective_id,
        )
        self.assertEqual(
            responder.call_args.args[0]["suggestion_learning_objective"][
                "objective_id"
            ],
            expected_objective_id,
        )

    @patch("learning.pedagogical_controller.generate_educational_response")
    def test_persisted_completion_selects_next_incomplete_objective_in_order(
        self, responder
    ):
        question = "How can I stay calm during exams?"
        selected = process_student_question(
            question,
            self.knowledge_base,
            self.keyword_map,
            self.progress_structure,
        )
        completed_objective = selected["learning_objective"]["objective_id"]
        chapter_id = selected["learning_objective"]["skill_id"]
        chapter = next(
            item
            for item in self.progress_structure["chapters"]
            if item["chapter_id"] == chapter_id
        )
        objective = next(
            item
            for item in chapter["objectives"]
            if item["objective_id"] == completed_objective
        )
        required_item_ids = set(objective.get("required_item_ids", []))

        def mongo_unavailable(*_args, **_kwargs):
            raise RuntimeError("Use isolated local progress storage in this test.")

        with patch("learning.interaction_store.MongoClient", mongo_unavailable):
            store = LearningStore()
        for item_id in required_item_ids:
            store.complete_item(
                "student-1",
                chapter_id,
                item_id,
                objective_id=completed_objective,
                required_item_ids=required_item_ids,
                source="chatbot",
            )

        persisted_progress = store.get_all_progress("student-1")
        self.assertIn(
            completed_objective,
            persisted_progress[0]["completed_objective_ids"],
        )
        next_objective = next(
            item
            for item in chapter["objectives"][
                next(
                    index
                    for index, item in enumerate(chapter["objectives"])
                    if item["objective_id"] == completed_objective
                ) + 1:
            ]
            if any(
                node.get("type") == "content"
                and node.get("skill_id") == chapter_id
                and node.get("objective_id") == item["objective_id"]
                for node in self.knowledge_base
            )
        )
        responder.return_value = {
            "answer": "Next objective answer.",
            "steps": [],
            "suggested_questions": [],
        }

        result = answer_student_question(
            question,
            self.knowledge_base,
            self.keyword_map,
            progress_documents=persisted_progress,
            progress_structure=self.progress_structure,
        )

        selected_after_completion = result["learning_context"]["learning_objective"]
        self.assertEqual(
            selected_after_completion["objective_id"],
            next_objective["objective_id"],
        )
        self.assertNotEqual(
            selected_after_completion["objective_id"],
            completed_objective,
        )
        self.assertEqual(
            responder.call_args.args[0]["suggestion_learning_objective"][
                "objective_id"
            ],
            next_objective["objective_id"],
        )


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "backend"))
    unittest.main()

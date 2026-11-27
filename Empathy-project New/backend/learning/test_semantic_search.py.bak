import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from learning.pedagogical_controller import process_student_question
from learning.semantic_search import semantic_search


def load_fixture(name):
    with (ROOT / "output" / name).open(encoding="utf-8") as file:
        return json.load(file)


class SemanticSearchTests(unittest.TestCase):
    def test_semantic_search_returns_relevant_content_nodes(self):
        knowledge_base = load_fixture("knowledge_base.json")
        results = semantic_search("How can I calm down before exams?", knowledge_base, limit=3)
        self.assertTrue(results)
        self.assertLessEqual(len(results), 3)
        self.assertTrue(all(item["id"] for item in results))
        self.assertTrue(all(item["type"] == "content" for item in results))

    def test_semantic_search_skips_empty_or_unclear_input(self):
        knowledge_base = load_fixture("knowledge_base.json")
        self.assertEqual(semantic_search("", knowledge_base), [])
        self.assertEqual(semantic_search(None, knowledge_base), [])

    def test_semantic_search_prefers_skill_scope_when_given(self):
        knowledge_base = load_fixture("knowledge_base.json")
        results = semantic_search("How can I calm down before exams?", knowledge_base, limit=3, skill_id="skill_01")
        self.assertTrue(results)
        self.assertTrue(all(item.get("skill_id") in {"skill_01", None} for item in results))

    def test_emotional_awareness_question_selects_emotional_awareness_objective(self):
        knowledge_base = load_fixture("knowledge_base.json")
        highlighted_keywords = load_fixture("highlighted_keywords.json")
        question = (
            'What is Emotional Awareness, and what are the key stages of the "Spark to Forest Fire" '
            'model used to understand how emotions develop and influence our actions?'
        )

        result = process_student_question(question, knowledge_base, highlighted_keywords)

        self.assertEqual(result["skill"]["id"], "node_0094")
        self.assertEqual(result["learning_objective"]["objective_id"], "obj_03_02")
        self.assertEqual(
            result["learning_objective"]["title"],
            "4. Identify the nature of your thoughts.",
        )


if __name__ == "__main__":
    unittest.main()

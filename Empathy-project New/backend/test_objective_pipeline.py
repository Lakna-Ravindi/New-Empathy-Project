import unittest

from app import resolve_parent_id, split_combined_objective_blocks, update_stack


class ObjectivePipelineTests(unittest.TestCase):
    def test_splits_combined_objective_description_after_skill_heading(self):
        blocks = [
            {"text": "Module Learning Objectives"},
            {"text": "By the end of this skill, you will be able to:"},
            {"text": "1.\u200b Name sensations. 2.\u200b Define the resilient zone. 3.\u200b Explain the nervous system."},
            {"text": "Introduction & Foundational Framework"},
        ]

        result = split_combined_objective_blocks(blocks)

        self.assertEqual(
            [block["text"] for block in result[2:5]],
            ["1.\u200b Name sensations.", "2.\u200b Define the resilient zone.", "3.\u200b Explain the nervous system."],
        )
        self.assertEqual(result[5]["text"], "Introduction & Foundational Framework")

    def test_active_objective_owns_following_topic_and_content(self):
        stack = {
            "chapter": "chapter",
            "module": None,
            "topic": "intro",
            "objective": None,
            "practice": None,
            "activity": None,
            "reflection": None,
            "assessment": None,
        }

        self.assertEqual(resolve_parent_id("learning_objective", stack), "intro")
        update_stack(stack, "learning_objective", "objective-1")

        self.assertIsNone(stack["topic"])
        self.assertEqual(resolve_parent_id("content", stack), "objective-1")
        self.assertEqual(resolve_parent_id("topic", stack), "objective-1")

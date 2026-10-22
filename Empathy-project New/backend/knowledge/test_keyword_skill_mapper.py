import unittest

from knowledge.keyword_skill_mapper import map_keywords_to_skills


class KeywordSkillMappingTests(unittest.TestCase):
    def setUp(self):
        self.nodes = [
            {
                "id": "skill-1",
                "type": "chapter",
                "title": "Skill 1",
                "content": "Skill 1",
                "parent_id": None,
            },
            {
                "id": "objective-1",
                "type": "learning_objective",
                "title": "Grounding objective",
                "content": "Grounding objective",
                "page": 1,
                "parent_id": "skill-1",
                "objective_id": "obj_01_01",
            },
            {
                "id": "content-1",
                "type": "content",
                "title": "Grounding",
                "content": "Grounding helps regulate the body.",
                "page": 1,
                "parent_id": "objective-1",
                "objective_id": "obj_01_01",
            },
        ]

    def test_candidate_maps_to_skill_and_objective(self):
        result = map_keywords_to_skills([{
            "text": "Grounding",
            "page": 1,
            "type": "keyword_candidate",
            "source": {
                "extraction_method": "source_concept",
                "node_id": "content-1",
            },
        }], self.nodes)

        self.assertEqual(result[0]["mapping_status"], "mapped")
        self.assertEqual(result[0]["skill_id"], "skill-1")
        self.assertEqual(result[0]["objective_id"], "obj_01_01")

    def test_candidate_without_objective_is_review(self):
        self.nodes[-1]["objective_id"] = None
        result = map_keywords_to_skills([{
            "text": "Grounding",
            "page": 1,
            "type": "keyword_candidate",
            "source": {"node_id": "content-1"},
        }], self.nodes)

        self.assertEqual(result[0]["mapping_status"], "review")

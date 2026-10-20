import unittest

from knowledge.objective_mapper import map_nodes_to_objectives


SKILLS = [{
    "skill_id": "skill_01",
    "skill_title": "Calming the Body and Mind",
    "key_concepts": [
        "Body Literacy",
        "Resilient Zone",
        "Sympathetic Nervous System",
        "Tracking",
        "Grounding",
        "Resourcing",
    ],
    "objectives": [
        {"objective_id": "obj_01_01", "objective_title": "Name body sensations"},
        {"objective_id": "obj_01_02", "objective_title": "Define the resilient zone"},
        {"objective_id": "obj_01_03", "objective_title": "Explain the nervous system"},
        {"objective_id": "obj_01_04", "objective_title": "Apply tracking grounding and resourcing"},
    ],
}]


class ObjectiveMapperTests(unittest.TestCase):
    def test_skill_one_heading_context_maps_content_until_next_heading(self):
        nodes = [
            {"id": "chapter", "type": "chapter", "title": "Skill 1: Calming the Body and Mind", "content": "", "parent_id": None},
            {"id": "heading-1", "type": "learning_objective", "title": "Objective 1: Name body sensations", "content": "", "parent_id": "chapter"},
            {"id": "body", "type": "content", "title": "Sensation vocabulary", "content": "Name physical sensations.", "parent_id": "heading-1"},
            {"id": "heading-3", "type": "learning_objective", "title": "Objective 3: Define the resilient zone", "content": "", "parent_id": "chapter"},
            {"id": "zone", "type": "content", "title": "Resilient zone", "content": "Recognize hyper-arousal and hypo-arousal.", "parent_id": "heading-3"},
            {"id": "heading-6", "type": "learning_objective", "title": "Objective 6: Apply tools", "content": "", "parent_id": "chapter"},
            {"id": "tracking", "type": "content", "title": "Tracking definition", "content": "Tracking observes sensations.", "parent_id": "heading-6"},
            {"id": "grounding", "type": "content", "title": "Grounding activity", "content": "Notice contact with the floor.", "parent_id": "heading-6"},
            {"id": "resourcing", "type": "content", "title": "Resource benefits", "content": "Build a supportive resource.", "parent_id": "heading-6"},
        ]

        mapped, _ = map_nodes_to_objectives(nodes, SKILLS)

        by_id = {node["id"]: node for node in mapped}
        self.assertEqual(by_id["body"]["objective_id"], "obj_01_01")
        self.assertEqual(by_id["zone"]["objective_id"], "obj_01_02")
        for node_id in ("tracking", "grounding", "resourcing"):
            self.assertEqual(by_id[node_id]["objective_id"], "obj_01_04")

    def test_unknown_skill_one_content_remains_review_or_unassigned(self):
        nodes = [
            {"id": "chapter", "type": "chapter", "title": "Skill 1: Calming the Body and Mind", "content": "", "parent_id": None},
            {"id": "content", "type": "content", "title": "Unclear example", "content": "A general story.", "parent_id": "chapter"},
        ]

        mapped, _ = map_nodes_to_objectives(nodes, SKILLS)

        self.assertIn(mapped[1]["match_status"], {"review", "unassigned"})
        self.assertIsNone(mapped[1]["objective_id"])


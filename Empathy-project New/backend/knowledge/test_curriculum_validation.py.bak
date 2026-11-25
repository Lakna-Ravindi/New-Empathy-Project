import json
from pathlib import Path

from knowledge.curriculum_validation import validate_curriculum


ROOT = Path(__file__).resolve().parents[2]


def test_curriculum_uses_canonical_seek_ids():
    report = validate_curriculum(ROOT / "output")

    assert report["status"] == "valid"
    assert report["knowledge_base"]["invalid_node_ids"] == []
    assert report["progress_structure"]["invalid_objectives"] == []


def test_validation_reports_missing_actionable_content_without_inventing_it():
    report = validate_curriculum(ROOT / "output")

    assert len(report["progress_structure"]["objectives_without_actionable_items"]) == 23
    assert report["content_skills"]["out_of_scope_skill_ids"] == [
        "skill_05", "skill_06", "skill_07", "skill_08"
    ]

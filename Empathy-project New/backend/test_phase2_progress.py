import json
from pathlib import Path

from knowledge.curriculum_validation import validate_curriculum
from learning.interaction_store import LearningStore
from learning.progress_service import ProgressService


ROOT = Path(__file__).resolve().parents[1]


def _objective():
    return {
        "objective_id": "objective-test",
        "items": [
            {"item_id": "lesson-1", "type": "text"},
            {"item_id": "lesson-2", "type": "video"},
            {"item_id": "activity-1", "type": "activity"},
            {"item_id": "quiz-1", "type": "quiz", "passing_score": 80},
        ],
    }


def test_runtime_structure_keeps_only_canonical_skills_and_objectives():
    structure = json.loads(
        (ROOT / "output" / "progress_structure.json").read_text(
            encoding="utf-8"
        )
    )
    service = ProgressService(structure=structure)

    assert [chapter["chapter_id"] for chapter in service.structure["chapters"]] == [
        "skill_01", "skill_02", "skill_03", "skill_04"
    ]
    assert sum(
        len(chapter["objectives"]) for chapter in service.structure["chapters"]
    ) == 24


def test_objective_validation_reports_all_canonical_objectives():
    report = validate_curriculum(ROOT / "output")

    assert len(report["objective_validation"]) == 24
    assert all(
        item["mapping_status"] == "matched"
        for item in report["objective_validation"]
    )
    assert len(report["progress_structure"]["objectives_without_actionable_items"]) == 23


def test_activity_eligibility_has_49_and_50_percent_boundaries():
    objective = _objective()
    below = ProgressService.completion_state(
        objective,
        ["lesson-1"],
        {},
    )
    at_threshold = ProgressService.completion_state(
        objective,
        ["lesson-1", "lesson-2"],
        {},
    )

    assert below["coverage_percent"] == 25.0
    assert below["activity_eligible"] is False
    assert at_threshold["coverage_percent"] == 50.0
    assert at_threshold["activity_eligible"] is True


def test_quiz_requires_90_percent_and_rejects_invalid_scores():
    objective = _objective()
    requirements = ProgressService.completion_requirements(objective)

    assert requirements["quiz_passing_scores"] == {"quiz-1": 90}
    assert ProgressService.completion_state(
        objective, ["activity-1", "quiz-1"], {"quiz-1": 89}
    )["quiz_passed"] is False
    assert ProgressService.completion_state(
        objective, ["activity-1", "quiz-1"], {"quiz-1": 90}
    )["quiz_passed"] is True
    assert ProgressService.completion_state(
        objective, ["activity-1", "quiz-1"], {"quiz-1": True}
    )["quiz_passed"] is False


def test_mongo_completion_predicate_preserves_duplicate_safe_completion():
    requirements = {
        "activity_item_ids": ["activity-1"],
        "quiz_passing_scores": {"quiz-1": 90},
    }

    assert LearningStore._objective_requirements_met(
        requirements,
        ["activity-1", "quiz-1"],
        {"quiz-1": 90},
    )
    assert not LearningStore._objective_requirements_met(
        requirements,
        ["activity-1", "quiz-1"],
        {"quiz-1": 89},
    )

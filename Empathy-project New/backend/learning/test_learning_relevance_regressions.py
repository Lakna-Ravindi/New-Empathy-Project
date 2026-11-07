import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from learning.gemini_responder import (
    _fallback_response,
    _fallback_suggested_questions,
    generate_educational_response,
)
from learning.pedagogical_controller import (
    _suggestion_target_for_progress,
    process_student_question,
)


def _fixture(name):
    with (ROOT / "output" / name).open(encoding="utf-8") as file:
        return json.load(file)


def test_beneficial_mental_states_uses_matching_objective_content():
    context = process_student_question(
        "What are beneficial mental states?",
        _fixture("knowledge_base.json"),
        _fixture("keyword_skill_map.json"),
        _fixture("progress_structure.json"),
    )

    assert context["status"] == "learning_path_selected"
    assert context["learning_objective"]["objective_id"] == "obj_03_06"
    assert context["question_relevant_content"]
    assert all(
        item["objective_id"] == "obj_03_06"
        for item in context["question_relevant_content"]
    )


def test_car_engine_maintenance_does_not_match_mind_body_analogy():
    context = process_student_question(
        "Tell me about car engine maintenance.",
        _fixture("knowledge_base.json"),
        _fixture("keyword_skill_map.json"),
        _fixture("progress_structure.json"),
    )

    assert context["status"] == "no_match"
    assert context["learning_objective"] is None
    assert context["recommended_activity"] is None
    assert context["question_relevant_content"] == []


@pytest.mark.parametrize(
    ("question", "objective_id", "required_text"),
    [
        ("What is body literacy?", "obj_01_01", "body literacy"),
        ("What is self-compassion?", "obj_04_01", "self-compassion"),
    ],
)
def test_definition_questions_use_complete_relevant_content(
    question, objective_id, required_text
):
    context = process_student_question(
        question,
        _fixture("knowledge_base.json"),
        _fixture("keyword_skill_map.json"),
        _fixture("progress_structure.json"),
    )

    assert context["learning_objective"]["objective_id"] == objective_id
    contents = [item["content"] for item in context["question_relevant_content"]]
    assert len(contents) == len({content.casefold() for content in contents})
    assert all(not content.endswith("invol") for content in contents)
    assert any(required_text in content.casefold() for content in contents)


def test_self_compassion_practice_uses_only_its_approved_activity():
    context = process_student_question(
        "How can I practice self-compassion?",
        _fixture("knowledge_base.json"),
        _fixture("keyword_skill_map.json"),
        _fixture("progress_structure.json"),
    )

    assert context["learning_objective"]["objective_id"] == "obj_04_04"
    assert context["recommended_activity"]["id"] == "node_0194"
    assert context["recommended_activity"]["objective_id"] == "obj_04_04"


def test_affective_empathy_question_stays_on_empathic_concern():
    context = process_student_question(
        "How are affective and cognitive empathy different?",
        _fixture("knowledge_base.json"),
        _fixture("keyword_skill_map.json"),
        _fixture("progress_structure.json"),
    )

    assert context["skill"]["id"] == "skill_07"
    assert context["learning_objective"]["objective_id"] == "obj_07_01"
    assert {
        item["id"] for item in context["question_relevant_content"]
    } == {"content_07_01"}


def test_repeated_self_compassion_question_does_not_switch_objective():
    first = process_student_question(
        "What is self-compassion?",
        _fixture("knowledge_base.json"),
        _fixture("keyword_skill_map.json"),
        _fixture("progress_structure.json"),
    )
    repeated = process_student_question(
        "What is self-compassion?",
        _fixture("knowledge_base.json"),
        _fixture("keyword_skill_map.json"),
        _fixture("progress_structure.json"),
    )

    assert first["learning_objective"]["objective_id"] == "obj_04_01"
    assert repeated["learning_objective"]["objective_id"] == "obj_04_01"
    assert repeated["skill"]["id"] == "skill_04"


def test_tracking_label_is_removed_without_truncating_node_0194():
    context = process_student_question(
        "How can I practice self-compassion?",
        _fixture("knowledge_base.json"),
        _fixture("keyword_skill_map.json"),
        _fixture("progress_structure.json"),
    )
    activity = context["recommended_activity"]
    assert "Tracking Instruction:" not in activity["title"]
    assert "Tracking Instruction:" not in activity["content"]
    assert (
        'ask yourself: "Is this an aspiration or an expectation?" '
        "and note down your feelings."
    ) in activity["content"]


def test_missing_approved_content_does_not_start_unrelated_learning():
    context = process_student_question(
        "What is empathy?",
        [
            {
                "id": "skill-a",
                "type": "chapter",
                "title": "Skill A",
                "parent_id": None,
            },
            {
                "id": "objective-a",
                "type": "learning_objective",
                "title": "Understand empathy",
                "content": "Understand empathy",
                "parent_id": "skill-a",
                "objective_id": "objective-a",
            },
            {
                "id": "content-a",
                "type": "content",
                "title": "A different topic",
                "content": "This content is about gardening.",
                "parent_id": "skill-a",
                "objective_id": "objective-a",
            },
        ],
        [],
    )

    assert context["status"] == "no_match"
    assert context["learning_objective"] is None
    assert context["recommended_activity"] is None
    assert context["question_relevant_content"] == []


def test_fallback_activity_is_student_facing_and_check_tests_taught_content():
    response = _fallback_response({
        "student_question": "How can I practise this?",
        "learning_objective": {"title": "An internal objective title"},
        "recommended_activity": {
            "id": "activity-1",
            "title": "Short practice",
            "type": "activity",
            "interaction_type": "practical",
            "content": "Create one short, optional practice from internal instructions.",
        },
        "question_relevant_content": [{
            "title": "Beneficial states",
            "content": "Beneficial mental states lead toward happiness and away from suffering.",
        }],
    })

    activity_text = response["steps"][2]["content"]
    check_text = response["steps"][3]["question"]
    assert "Create one short" not in activity_text
    assert "Beneficial mental states" in check_text
    assert "internal objective title" not in check_text
    assert "connects to the approved SEEK material" not in check_text


def test_fallback_response_does_not_repeat_paragraphs():
    response = _fallback_response({
        "student_question": "What does this mean?",
        "learning_objective": None,
        "recommended_activity": None,
        "question_relevant_content": [{
            "title": "Concept",
            "content": "A clear explanation of the concept.",
        }],
    })

    paragraphs = [
        paragraph.strip().casefold()
        for paragraph in response["answer"].split("\n\n")
        if paragraph.strip()
    ]
    assert len(paragraphs) == len(set(paragraphs))


def test_missing_activity_has_student_facing_unavailable_message():
    response = _fallback_response({
        "student_question": "What is body literacy?",
        "learning_objective": {"title": "Body literacy"},
        "recommended_activity": None,
        "question_relevant_content": [{
            "title": "Body literacy",
            "content": "Body literacy is the ability to understand body sensations.",
        }],
    })

    assert response["steps"][2]["activity"] is None
    assert "No activity was selected" in response["steps"][2]["content"]


def test_next_incomplete_objective_supplies_suggestions_without_activity():
    current = {
        "skill": {"id": "skill-test"},
        "learning_objective": {
            "id": "objective-4",
            "objective_id": "objective-4",
            "skill_id": "skill-test",
        },
    }
    structure = {
        "chapters": [{
            "chapter_id": "skill-test",
            "objectives": [
                {"objective_id": "objective-4", "title": "Completed"},
                {
                    "objective_id": "objective-5",
                    "title": (
                        "Define empathy and differentiate affective empathy "
                        "from cognitive empathy."
                    ),
                    "items": [],
                },
            ],
        }],
    }
    content = [{
        "id": "content-5",
        "type": "content",
        "skill_id": "skill-test",
        "objective_id": "objective-5",
        "title": "Two forms of empathy",
        "content": (
            "Affective empathy concerns shared feelings. Cognitive empathy "
            "concerns understanding another person's perspective."
        ),
    }]
    target = _suggestion_target_for_progress(
        current,
        content,
        [{"skill_id": "skill-test", "completed_objective_ids": ["objective-4"]}],
        structure,
    )
    assert target["learning_objective"]["objective_id"] == "objective-5"
    assert target["recommended_activity"] is None
    questions = _fallback_suggested_questions({
        **current,
        "suggestion_learning_objective": target["learning_objective"],
        "suggestion_relevant_content": target["question_relevant_content"],
    })
    assert 3 <= len(questions) <= 4
    assert all("objective-5" not in question for question in questions)


def test_suggestions_are_present_with_or_without_an_activity():
    suggestion_context = {
        "suggestion_learning_objective": {
            "objective_id": "objective-5",
            "title": "Understand empathy",
        },
        "suggestion_relevant_content": [{
            "title": "Empathy",
            "content": (
                "Affective empathy concerns shared feelings. Cognitive empathy "
                "concerns understanding another person's perspective."
            ),
        }],
    }
    for activity in (None, {
        "id": "activity-5",
        "title": "Practice",
        "type": "activity",
        "content": "Notice one perspective in a conversation.",
    }):
        response = _fallback_response({
            "student_question": "What should I learn next?",
            "learning_objective": suggestion_context["suggestion_learning_objective"],
            "recommended_activity": activity,
            "question_relevant_content": [],
            **suggestion_context,
        })
        assert 3 <= len(response["suggested_questions"]) <= 4
        if activity is None:
            assert response["steps"][2]["activity"] is None
        else:
            assert response["steps"][2]["activity"]["id"] == "activity-5"


def test_duplicate_content_is_removed_before_gemini_prompt(monkeypatch):
    captured = {}

    class FakeModels:
        def generate_content(self, **kwargs):
            captured["prompt"] = kwargs["contents"]
            return type(
                "Response",
                (),
                {"text": '{"answer":"truncated"', "candidates": []},
            )()

    class FakeClient:
        models = FakeModels()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        "learning.gemini_responder.genai.Client",
        lambda **kwargs: FakeClient(),
    )
    response = generate_educational_response({
        "status": "learning_path_selected",
        "student_question": "What is body literacy?",
        "skill": {"id": "skill_01", "title": "Body literacy"},
        "learning_objective": {"id": "obj_01_01", "title": "Body literacy"},
        "recommended_activity": None,
        "question_relevant_content": [
            {"id": "content-a", "content": "Body literacy is awareness."},
            {"id": "content-b", "content": "Body literacy is awareness."},
        ],
    })

    assert response["steps"]
    assert captured["prompt"].count("Body literacy is awareness.") == 1

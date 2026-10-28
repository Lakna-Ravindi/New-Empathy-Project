import copy

import server
from learning.pedagogical_controller import process_student_question
from learning.progress_service import ProgressService


def _structure():
    return {
        "chapters": [{
            "chapter_id": "skill_01",
            "objectives": [{
                "objective_id": "obj_01_04",
                "title": "Apply tracking, grounding, and resourcing techniques",
                "required_item_ids": ["activity_01_04"],
                "items": [{
                    "item_id": "activity_01_04",
                    "type": "activity",
                    "title": "Contact Grounding Practice",
                    "content": "",
                }],
            }],
        }],
    }


def _multi_item_structure():
    items = [
        {"item_id": "content_test", "type": "text", "title": "Read the lesson"},
        {"item_id": "activity_test", "type": "activity", "title": "Try the activity"},
        {"item_id": "video_test", "type": "video", "title": "Watch the video"},
        {"item_id": "quiz_test", "type": "quiz", "title": "Pass the quiz"},
    ]
    return {
        "chapters": [{
            "chapter_id": "skill_test",
            "title": "Test skill",
            "objectives": [{
                "objective_id": "objective_test",
                "title": "Test objective",
                "required_item_ids": [item["item_id"] for item in items],
                "items": items,
            }],
        }],
    }


def _objective_rule_structure():
    return {
        "chapters": [{
            "chapter_id": "skill_test",
            "title": "Test skill",
            "objectives": [
                {
                    "objective_id": "objective_test",
                    "title": "First objective",
                    "required_item_ids": [
                        "content_test",
                        "activity_test",
                        "video_test",
                        "quiz_test",
                    ],
                    "items": [
                        {"item_id": "content_test", "type": "text", "title": "Read"},
                        {"item_id": "activity_test", "type": "activity", "title": "Reflect"},
                        {"item_id": "video_test", "type": "video", "title": "Watch"},
                        {
                            "item_id": "quiz_test",
                            "type": "quiz",
                            "title": "Check your understanding",
                            "passing_score": 75,
                            "questions": [{"question": "Q"}],
                        },
                    ],
                },
                {
                    "objective_id": "objective_next",
                    "title": "Next objective",
                    "items": [
                        {"item_id": "activity_next", "type": "activity"},
                        {
                            "item_id": "quiz_next",
                            "type": "quiz",
                            "passing_score": 80,
                        },
                    ],
                },
            ],
        }],
    }


class StoreStub:
    def __init__(self):
        self.progress = {
            "student_id": "student-1",
            "skill_id": "skill_01",
            "completed_item_ids": [],
            "completed_objective_ids": [],
            "quiz_scores": {},
            "quiz_attempts": [],
        }
        self.complete_calls = []
        self.saved_interactions = {}

    def get_progress(self, student_id, skill_id):
        return copy.deepcopy(self.progress)

    def get_interaction(self, student_id, interaction_id):
        return self.saved_interactions.get(interaction_id)

    def get_all_progress(self, student_id):
        return [copy.deepcopy(self.progress)]

    def complete_item(self, student_id, skill_id, item_id, **kwargs):
        self.complete_calls.append((student_id, skill_id, item_id, kwargs))
        self.progress["skill_id"] = skill_id
        if item_id not in self.progress["completed_item_ids"]:
            self.progress["completed_item_ids"].append(item_id)
        if kwargs.get("quiz_score") is not None:
            old_score = self.progress["quiz_scores"].get(item_id, 0)
            self.progress["quiz_scores"][item_id] = max(
                old_score,
                kwargs["quiz_score"],
            )
            self.progress["quiz_attempts"].append({
                "item_id": item_id,
                "score": kwargs["quiz_score"],
            })
        requirements = kwargs.get("objective_completion_requirements") or {}
        activities_done = bool(
            set(requirements.get("activity_item_ids", []))
            & set(self.progress["completed_item_ids"])
        )
        quizzes_done = any(
            quiz_id in self.progress["completed_item_ids"]
            and self.progress["quiz_scores"].get(quiz_id, 0) >= threshold
            for quiz_id, threshold in requirements.get(
                "quiz_passing_scores",
                {},
            ).items()
        )
        if (
            kwargs.get("objective_id")
            and activities_done
            and quizzes_done
            and kwargs["objective_id"] not in self.progress["completed_objective_ids"]
        ):
            self.progress["completed_objective_ids"].append(kwargs["objective_id"])
        return copy.deepcopy(self.progress)

    def record_quiz_attempt(
        self,
        student_id,
        skill_id,
        item_id,
        score,
        **kwargs,
    ):
        old_score = self.progress["quiz_scores"].get(item_id, 0)
        self.progress["quiz_scores"][item_id] = max(old_score, score)
        self.progress["quiz_attempts"].append({
            "item_id": item_id,
            "score": score,
        })
        return copy.deepcopy(self.progress)

    def complete_objective(self, student_id, skill_id, objective, next_objective):
        self.progress["skill_id"] = skill_id
        objective_id = objective["id"]
        if objective_id not in self.progress["completed_objective_ids"]:
            self.progress["completed_objective_ids"].append(objective_id)
        self.progress["next_recommended_learning_objective"] = next_objective
        return copy.deepcopy(self.progress)


def call_completion(client, body, store, structure=None):
    original_structure = server.progress_service.structure
    original_store = server.learning_store
    server.progress_service.structure = structure or _structure()
    server.learning_store = store
    try:
        with client.application.test_request_context(
            "/api/progress/item-complete",
            method="POST",
            json=body,
        ):
            server.g.student_id = "student-1"
            return server.complete_learning_item.__wrapped__()
    finally:
        server.progress_service.structure = original_structure
        server.learning_store = original_store


def call_objective_completion(client, objective_id, body, store, structure=None):
    original_structure = server.progress_service.structure
    original_store = server.learning_store
    server.progress_service.structure = structure or _objective_rule_structure()
    server.learning_store = store
    try:
        with client.application.test_request_context(
            f"/api/objectives/{objective_id}/complete",
            method="POST",
            json=body,
        ):
            server.g.student_id = "student-1"
            return server.complete_objective.__wrapped__(objective_id)
    finally:
        server.progress_service.structure = original_structure
        server.learning_store = original_store


def test_completion_resolves_authored_grounding_item_and_updates_progress():
    store = StoreStub()
    response, status = call_completion(
        server.app.test_client(),
        {
            "skill_id": "skill_01",
            "item_id": "activity_01_04",
            "item_type": "activity",
            "source": "chatbot",
        },
        store,
    )

    assert status == 200
    assert store.complete_calls[0][2] == "activity_01_04"
    assert response.get_json()["completed_item_ids"] == ["activity_01_04"]


def test_activity_completion_keeps_objective_incomplete_until_quiz_is_passed():
    structure = _multi_item_structure()
    store = StoreStub()

    response, status = call_completion(
        server.app.test_client(),
        {
            "skill_id": "skill_test",
            "item_id": "activity_test",
            "item_type": "activity",
            "source": "chatbot",
        },
        store,
        structure,
    )

    assert status == 200
    progress = response.get_json()
    assert progress["completed_item_ids"] == ["activity_test"]
    assert progress["completed_objective_ids"] == []
    summary = ProgressService(structure=structure).calculate([progress])
    objective = summary["skills"][0]["objectives"][0]
    assert objective["missing_required_item_ids"] == ["quiz_test"]


def test_objective_completes_after_activity_and_quiz_without_content_or_video():
    structure = _multi_item_structure()
    store = StoreStub()
    client = server.app.test_client()
    for item_id, item_type in (
        ("activity_test", "activity"),
        ("quiz_test", "quiz"),
    ):
        body = {
            "skill_id": "skill_test",
            "item_id": item_id,
            "item_type": item_type,
            "source": "chatbot",
        }
        if item_type == "quiz":
            body["quiz_score"] = 90
        response, status = call_completion(client, body, store, structure)
        assert status == 200

    progress = response.get_json()
    assert progress["completed_item_ids"] == ["activity_test", "quiz_test"]
    assert progress["completed_objective_ids"] == ["objective_test"]
    summary = ProgressService(structure=structure).calculate([progress])
    objective = summary["skills"][0]["objectives"][0]
    assert objective["completed"] is True
    assert objective["missing_required_item_ids"] == []


def test_chatbot_quick_check_completes_objective_and_updates_next_recommendation():
    structure = _objective_rule_structure()
    store = StoreStub()
    store.saved_interactions["interaction-1"] = {
        "selected_skill": {"id": "skill_test"},
        "learning_objective": {"objective_id": "objective_test"},
        "learning_steps": [{
            "type": "practice_check",
            "correct_answer": True,
        }],
    }
    store.progress.update({
        "skill_id": "skill_test",
        "completed_item_ids": ["activity_test"],
    })

    response, status = call_objective_completion(
        server.app.test_client(),
        "objective_test",
        {
            "skill_id": "skill_test",
            "source": "chatbot",
            "interaction_id": "interaction-1",
            "quick_check_answer": True,
        },
        store,
        structure,
    )

    assert status == 200
    progress = response.get_json()
    assert progress["completed_objective_ids"] == ["objective_test"]
    assert progress["next_recommended_learning_objective"]["id"] == "objective_next"
    summary = ProgressService(structure=structure).calculate([progress])
    current_objective = summary["skills"][0]["objectives"][0]
    assert current_objective["completed"] is True
    assert current_objective["quiz_passed"] is False


def test_chatbot_quick_check_cannot_complete_objective_without_saved_activity():
    store = StoreStub()
    store.saved_interactions["interaction-1"] = {
        "selected_skill": {"id": "skill_test"},
        "learning_objective": {"objective_id": "objective_test"},
        "learning_steps": [{
            "type": "practice_check",
            "correct_answer": True,
        }],
    }

    response, status = call_objective_completion(
        server.app.test_client(),
        "objective_test",
        {
            "skill_id": "skill_test",
            "source": "chatbot",
            "interaction_id": "interaction-1",
            "quick_check_answer": True,
        },
        store,
        _objective_rule_structure(),
    )

    assert status == 422
    assert "activity" in response.get_json()["missing_completion_conditions"]
    assert store.progress["completed_objective_ids"] == []


def test_quick_check_completion_requires_chatbot_source_and_correct_answer():
    store = StoreStub()
    store.saved_interactions["interaction-1"] = {
        "selected_skill": {"id": "skill_test"},
        "learning_objective": {"objective_id": "objective_test"},
        "learning_steps": [{
            "type": "practice_check",
            "correct_answer": True,
        }],
    }
    store.progress.update({
        "skill_id": "skill_test",
        "completed_item_ids": ["activity_test"],
    })

    response, status = call_objective_completion(
        server.app.test_client(),
        "objective_test",
        {
            "skill_id": "skill_test",
            "source": "skills_page",
            "interaction_id": "interaction-1",
            "quick_check_answer": True,
        },
        store,
        _objective_rule_structure(),
    )

    assert status == 422
    assert store.progress["completed_objective_ids"] == []


def test_objective_completion_rejects_unverified_quick_check_answers():
    cases = (
        ("no_saved_interaction", None, True),
        ("missing_answer_key", {}, True),
        ("invalid_answer_key", {"correct_answer": "true"}, True),
        ("incorrect_answer", {"correct_answer": False}, True),
    )
    for name, answer_key, submitted_answer in cases:
        store = StoreStub()
        store.progress.update({
            "skill_id": "skill_test",
            "completed_item_ids": ["activity_test"],
        })
        interaction = {
            "selected_skill": {"id": "skill_test"},
            "learning_objective": {"objective_id": "objective_test"},
            "learning_steps": [{
                "type": "practice_check",
                **({"correct_answer": answer_key} if answer_key is not None else {}),
            }],
        }
        if name != "no_saved_interaction":
            store.saved_interactions["interaction-1"] = interaction

        response, status = call_objective_completion(
            server.app.test_client(),
            "objective_test",
            {
                "skill_id": "skill_test",
                "source": "chatbot",
                "quick_check_passed": True,
                "interaction_id": "interaction-1",
                "quick_check_answer": submitted_answer,
            },
            store,
            _objective_rule_structure(),
        )

        assert status == 422, name
        assert store.progress["completed_objective_ids"] == [], name


def test_skill_four_objective_reports_actual_missing_items_after_activity_only():
    structure = server.progress_service.get_structure()
    chapter = next(
        item for item in structure["chapters"]
        if item["chapter_id"] == "skill_04"
    )
    objective = next(
        item for item in chapter["objectives"]
        if item["objective_id"] == "obj_04_01"
    )
    progress_service = ProgressService(structure=structure)
    partial_progress = {
        "skill_id": "skill_04",
        "completed_item_ids": ["activity_04_01"],
        "completed_objective_ids": [],
    }

    partial = progress_service.calculate([partial_progress])
    summary = next(
        item for item in partial["skills"]
        if item["skill_id"] == "skill_04"
    )
    partial_objective = next(
        item for item in summary["objectives"]
        if item["objective_id"] == "obj_04_01"
    )
    assert partial_objective["missing_required_item_ids"] == ["quiz_04_01"]

    completed_progress = {
        **partial_progress,
        "completed_item_ids": ["activity_04_01", "quiz_04_01"],
        "quiz_scores": {"quiz_04_01": 80},
        "completed_objective_ids": ["obj_04_01"],
    }
    complete = progress_service.calculate([completed_progress])
    summary = next(
        item for item in complete["skills"]
        if item["skill_id"] == "skill_04"
    )
    complete_objective = next(
        item for item in summary["objectives"]
        if item["objective_id"] == "obj_04_01"
    )
    assert complete_objective["completed"] is True
    assert complete_objective["missing_required_item_ids"] == []


def test_failing_quiz_attempt_is_saved_without_completing_quiz_or_objective():
    store = StoreStub()
    response, status = call_completion(
        server.app.test_client(),
        {
            "skill_id": "skill_test",
            "item_id": "quiz_test",
            "item_type": "quiz",
            "quiz_score": 60,
        },
        store,
        _objective_rule_structure(),
    )

    assert status == 200
    result = response.get_json()
    assert result["quiz_passed"] is False
    assert result["objective_completed"] is False
    assert result["completed_item_ids"] == []
    assert result["quiz_scores"]["quiz_test"] == 60
    assert result["missing_completion_conditions"] == ["activity", "quiz"]
    assert {item["item_id"] for item in result["missing_completion_items"]} == {
        "activity_test",
        "quiz_test",
    }


def test_activity_complete_but_failed_quiz_keeps_objective_incomplete():
    store = StoreStub()
    client = server.app.test_client()
    structure = _objective_rule_structure()
    activity_response, status = call_completion(
        client,
        {
            "skill_id": "skill_test",
            "item_id": "activity_test",
            "item_type": "activity",
        },
        store,
        structure,
    )
    assert status == 200
    assert activity_response.get_json()["objective_completed"] is False

    quiz_response, status = call_completion(
        client,
        {
            "skill_id": "skill_test",
            "item_id": "quiz_test",
            "item_type": "quiz",
            "quiz_score": 60,
        },
        store,
        structure,
    )
    assert status == 200
    result = quiz_response.get_json()
    assert result["objective_completed"] is False
    assert result["missing_completion_conditions"] == ["quiz"]
    assert "quiz_test" not in result["completed_item_ids"]
    assert result["quiz_scores"]["quiz_test"] == 60


def test_passing_quiz_without_activity_does_not_complete_objective():
    store = StoreStub()
    response, status = call_completion(
        server.app.test_client(),
        {
            "skill_id": "skill_test",
            "item_id": "quiz_test",
            "item_type": "quiz",
            "quiz_score": 75,
        },
        store,
        _objective_rule_structure(),
    )

    assert status == 200
    result = response.get_json()
    assert result["quiz_passed"] is True
    assert result["objective_completed"] is False
    assert result["completed_item_ids"] == ["quiz_test"]
    assert result["missing_completion_conditions"] == ["activity"]
    assert result["missing_completion_items"][0]["title"] == "Reflect"


def test_completion_persists_across_requests_and_recommends_next_objective():
    store = StoreStub()
    client = server.app.test_client()
    structure = _objective_rule_structure()

    activity_response, status = call_completion(
        client,
        {
            "skill_id": "skill_test",
            "item_id": "activity_test",
            "item_type": "activity",
        },
        store,
        structure,
    )
    assert status == 200
    assert activity_response.get_json()["objective_completed"] is False

    quiz_response, status = call_completion(
        client,
        {
            "skill_id": "skill_test",
            "item_id": "quiz_test",
            "item_type": "quiz",
            "quiz_score": 75,
        },
        store,
        structure,
    )
    assert status == 200
    result = quiz_response.get_json()
    assert result["completed_item_ids"] == ["activity_test", "quiz_test"]
    assert result["completed_objective_ids"] == ["objective_test"]
    assert result["objective_completed"] is True
    assert result["completed_objective"] == {
        "id": "objective_test",
        "title": "First objective",
    }
    assert result["next_incomplete_objective"]["id"] == "objective_next"
    assert result["next_recommended_learning_objective"]["id"] == "objective_next"


def test_mismatched_or_unknown_curriculum_ids_cannot_complete_objectives():
    store = StoreStub()
    structure = _objective_rule_structure()
    response, status = call_completion(
        server.app.test_client(),
        {
            "skill_id": "skill_test",
            "item_id": "activity_test",
            "item_type": "activity",
            "objective_id": "objective_next",
        },
        store,
        structure,
    )

    assert status == 400
    assert store.progress["completed_item_ids"] == []

    response, status = call_completion(
        server.app.test_client(),
        {
            "skill_id": "skill_test",
            "item_id": "unknown_activity",
            "item_type": "activity",
            "objective_id": "objective_test",
        },
        store,
        structure,
    )
    assert status == 400
    assert store.progress["completed_objective_ids"] == []


def test_grounding_recommendation_gets_authored_progress_metadata():
    original_structure = server.progress_service.structure
    original_nodes = server.nodes_by_id
    server.progress_service.structure = _structure()
    server.nodes_by_id = {
        "node_grounding": {
            "id": "node_grounding",
            "type": "topic",
            "title": "Grounding Activity",
            "content": "Contact Grounding Practice",
            "parent_id": "node_objective",
        },
        "node_objective": {
            "id": "node_objective",
            "type": "learning_objective",
            "objective_id": "obj_01_04",
        }
    }
    context = {
        "skill": {"id": "skill_01"},
        "recommended_activity": {
            "id": "node_grounding",
            "type": "activity",
            "title": "Grounding Activity",
            "content": "Contact Grounding Practice",
        },
    }
    try:
        server.attach_progress_item_id(context)
    finally:
        server.progress_service.structure = original_structure
        server.nodes_by_id = original_nodes

    activity = context["recommended_activity"]
    assert activity["progress_item_id"] == "activity_01_04"
    assert activity["progress_objective_id"] == "obj_01_04"


def test_authored_item_id_is_used_as_direct_progress_reference():
    original_structure = server.progress_service.structure
    original_nodes = server.nodes_by_id
    server.progress_service.structure = _structure()
    server.nodes_by_id = {}
    context = {
        "skill": {"id": "skill_01"},
        "learning_objective": {
            "id": "obj_01_04",
            "objective_id": "obj_01_04",
        },
        "recommended_activity": {
            "id": "activity_01_04",
            "objective_id": "obj_01_04",
            "type": "activity",
            "title": "Authored activity title not copied from controller",
            "content": "This wording does not match the authored text.",
        },
    }
    try:
        server.attach_progress_item_id(context)
    finally:
        server.progress_service.structure = original_structure
        server.nodes_by_id = original_nodes

    activity = context["recommended_activity"]
    assert activity["progress_item_id"] == "activity_01_04"
    assert activity["progress_item_type"] == "activity"
    assert activity["progress_objective_id"] == "obj_01_04"


def test_skill_four_complete_action_receives_authored_activity_metadata():
    context = process_student_question(
        "What is self compassion?",
        server.knowledge_base,
        server.keyword_skill_map,
        server.progress_service.get_structure(),
    )

    assert context["skill"]["id"] == "node_0147"
    assert context["learning_objective"]["objective_id"] == "obj_04_01"
    assert context["recommended_activity"]["id"] == "activity_04_01"

    server.attach_progress_item_id(context)

    activity = context["recommended_activity"]
    assert activity["progress_item_id"] == "activity_04_01"
    assert activity["progress_item_type"] == "activity"
    assert activity["progress_objective_id"] == "obj_04_01"


def test_unresolved_activity_does_not_receive_fabricated_progress_metadata():
    original_structure = server.progress_service.structure
    original_nodes = server.nodes_by_id
    server.progress_service.structure = {
        "chapters": [{
            "chapter_id": "skill_01",
            "objectives": [{
                "objective_id": "obj_01_04",
                "items": [
                    {
                        "item_id": "activity-a",
                        "type": "activity",
                        "title": "A different activity",
                    },
                    {
                        "item_id": "activity-b",
                        "type": "activity",
                        "title": "Another different activity",
                    },
                ],
            }],
        }],
    }
    server.nodes_by_id = {}
    context = {
        "skill": {"id": "skill_01"},
        "learning_objective": {
            "id": "obj_01_04",
            "objective_id": "obj_01_04",
        },
        "recommended_activity": {
            "id": "unknown-activity",
            "objective_id": "obj_01_04",
            "type": "activity",
            "title": "Unmatched content",
            "content": "No curriculum reference matches this activity.",
        },
    }
    try:
        server.attach_progress_item_id(context)
    finally:
        server.progress_service.structure = original_structure
        server.nodes_by_id = original_nodes

    activity = context["recommended_activity"]
    assert "progress_item_id" not in activity
    assert "progress_item_type" not in activity
    assert "progress_objective_id" not in activity


def test_learning_response_keeps_authored_metadata_in_nested_context():
    original_structure = server.progress_service.structure
    original_nodes = server.nodes_by_id
    original_answer = server.answer_student_question
    original_store = server.learning_store
    server.progress_service.structure = _structure()
    server.nodes_by_id = {
        "node_grounding": {
            "id": "node_grounding",
            "type": "topic",
            "title": "Grounding Activity",
            "content": "Contact Grounding Practice",
            "parent_id": "node_objective",
        },
        "node_objective": {
            "id": "node_objective",
            "type": "learning_objective",
            "objective_id": "obj_01_04",
        },
    }

    class ResponseStore:
        saved_steps = None

        def get_all_progress(self, *_args):
            return []

        def save_interaction(self, *_args, learning_steps=None):
            self.saved_steps = learning_steps
            return "interaction-1"

        def record_learning_event(self, *_args, **_kwargs):
            return None

    context = {
        "student_question": "How can I practice grounding?",
        "status": "learning_path_selected",
        "skill": {"id": "skill_01"},
        "recommended_activity": {
            "id": "node_grounding",
            "type": "activity",
            "title": "Grounding Activity",
            "content": "Contact Grounding Practice",
        },
    }
    server.answer_student_question = lambda *_args, **_kwargs: {
        "learning_context": context,
        "answer": "Try this grounding activity.",
        "steps": [],
        "suggested_questions": ["How do I ground myself?", "What does grounding mean?", "When can I practice grounding?"],
    }
    response_store = ResponseStore()
    server.learning_store = response_store

    try:
        with server.app.test_request_context(
            "/api/learning-response",
            method="POST",
            json={"question": "How can I practice grounding?", "answer_style": "short"},
        ):
            server.g.student_id = "student-1"
            response, status = server.learning_response.__wrapped__()
        activity = response.get_json()["learning_context"]["recommended_activity"]
    finally:
        server.progress_service.structure = original_structure
        server.nodes_by_id = original_nodes
        server.answer_student_question = original_answer
        server.learning_store = original_store

    assert status == 200
    assert activity["progress_item_id"] == "activity_01_04"
    assert activity["progress_item_type"] == "activity"
    assert activity["progress_objective_id"] == "obj_01_04"
    assert len(response.get_json()["suggested_questions"]) == 3
    assert response_store.saved_steps == []


def test_real_objective_context_resolves_authored_activity_when_controller_has_no_node_activity():
    original_structure = server.progress_service.structure
    original_nodes = server.nodes_by_id
    server.progress_service.structure = _structure()
    server.nodes_by_id = {
        "node_0001": {
            "id": "node_0001",
            "type": "chapter",
            "title": "Skill 1",
        },
    }
    context = {
        "skill": {"id": "node_0001", "title": "Skill 1"},
        "learning_objective": {
            "id": "obj_01_04",
            "objective_id": "obj_01_04",
            "type": "learning_objective",
            "title": "Apply grounding",
        },
        "recommended_activity": None,
    }
    try:
        server.attach_progress_item_id(context)
    finally:
        server.progress_service.structure = original_structure
        server.nodes_by_id = original_nodes

    assert context["skill"]["id"] == "skill_01"
    assert context["recommended_activity"]["id"] == "activity_01_04"
    assert context["recommended_activity"]["progress_item_id"] == "activity_01_04"
    assert context["recommended_activity"]["progress_item_type"] == "activity"
    assert context["recommended_activity"]["progress_objective_id"] == "obj_01_04"


def test_invalid_item_id_is_rejected_without_progress_update():
    store = StoreStub()
    response, status = call_completion(
        server.app.test_client(),
        {
            "skill_id": "skill_01",
            "item_id": "node_0047",
            "item_type": "activity",
            "source": "chatbot",
        },
        store,
    )

    assert status == 400
    assert store.complete_calls == []
    assert store.progress["completed_item_ids"] == []


def test_item_type_mismatch_is_rejected_without_progress_update():
    store = StoreStub()
    response, status = call_completion(
        server.app.test_client(),
        {
            "skill_id": "skill_01",
            "item_id": "activity_01_04",
            "item_type": "quiz",
            "source": "chatbot",
        },
        store,
    )

    assert status == 400
    assert store.complete_calls == []

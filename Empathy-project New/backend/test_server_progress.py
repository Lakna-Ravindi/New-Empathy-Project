import copy

import server


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


class StoreStub:
    def __init__(self):
        self.progress = {
            "student_id": "student-1",
            "skill_id": "skill_01",
            "completed_item_ids": [],
            "completed_objective_ids": [],
        }
        self.complete_calls = []

    def get_progress(self, student_id, skill_id):
        return copy.deepcopy(self.progress)

    def complete_item(self, student_id, skill_id, item_id, **kwargs):
        self.complete_calls.append((student_id, skill_id, item_id, kwargs))
        self.progress["completed_item_ids"].append(item_id)
        self.progress["completed_objective_ids"].append(kwargs["objective_id"])
        return copy.deepcopy(self.progress)


def call_completion(client, body, store):
    original_structure = server.progress_service.structure
    original_store = server.learning_store
    server.progress_service.structure = _structure()
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
        def save_interaction(self, *_args):
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
    }
    server.learning_store = ResponseStore()

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

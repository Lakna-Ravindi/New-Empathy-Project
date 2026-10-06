import learning.interaction_store as module


class ConnectionErrorStub(Exception):
    pass


def test_learning_store_falls_back_when_mongo_is_unavailable(monkeypatch):
    def fake_mongo_client(*args, **kwargs):
        raise ConnectionErrorStub("database unavailable")

    monkeypatch.setattr(module, "MongoClient", fake_mongo_client)

    store = module.LearningStore()
    interaction_id = store.save_interaction(
        "student-1",
        {"student_question": "Why do I feel upset?", "status": "learning_path_selected"},
        "A calm explanation.",
    )

    assert interaction_id.startswith("local-")
    progress = store.get_progress("student-1", "skill-1")
    assert progress["completed_objective_ids"] == []


def test_learning_events_are_deduplicated_and_keep_sources(monkeypatch):
    monkeypatch.setattr(module, "MongoClient", lambda *args, **kwargs: (_ for _ in ()).throw(ConnectionErrorStub()))
    store = module.LearningStore()

    store.record_learning_event(
        "student-1",
        "skill-1",
        "content_01_01",
        objective_id="obj_01_01",
        source="chatbot",
        status="introduced",
    )
    store.record_learning_event(
        "student-1",
        "skill-1",
        "content_01_01",
        objective_id="obj_01_01",
        source="chatbot",
        status="introduced",
    )
    store.complete_item(
        "student-1",
        "skill-1",
        "content_01_01",
        objective_id="obj_01_01",
        required_item_ids={"content_01_01"},
        source="skills_page",
    )

    progress = store.get_progress("student-1", "skill-1")
    assert progress["completed_item_ids"] == ["content_01_01"]
    assert len(progress["learning_events"]) == 1
    event = progress["learning_events"][0]
    assert event["status"] == "completed"
    assert event["sources"] == ["chatbot", "skills_page"]

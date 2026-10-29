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


def test_quiz_attempts_are_saved_separately_and_objective_requires_activity_and_pass(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "MongoClient",
        lambda *args, **kwargs: (_ for _ in ()).throw(ConnectionErrorStub()),
    )
    store = module.LearningStore()
    requirements = {
        "activity_item_ids": ["activity_1"],
        "quiz_passing_scores": {"quiz_1": 75},
    }

    failed = store.record_quiz_attempt(
        "student-1",
        "skill-1",
        "quiz_1",
        60,
        objective_id="objective-1",
    )
    assert failed["quiz_scores"]["quiz_1"] == 60
    assert failed["completed_item_ids"] == []
    assert failed["completed_objective_ids"] == []

    store.complete_item(
        "student-1",
        "skill-1",
        "quiz_1",
        objective_id="objective-1",
        objective_completion_requirements=requirements,
        quiz_score=80,
    )
    quiz_only = store.get_progress("student-1", "skill-1")
    assert quiz_only["completed_item_ids"] == ["quiz_1"]
    assert quiz_only["completed_objective_ids"] == []
    assert quiz_only["quiz_scores"]["quiz_1"] == 80

    completed = store.complete_item(
        "student-1",
        "skill-1",
        "activity_1",
        objective_id="objective-1",
        objective_completion_requirements=requirements,
    )
    persisted = store.get_progress("student-1", "skill-1")
    assert completed["completed_objective_ids"] == ["objective-1"]
    assert persisted["completed_objective_ids"] == ["objective-1"]
    assert len(persisted["quiz_attempts"]) == 2

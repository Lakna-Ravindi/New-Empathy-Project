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

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest
from pymongo.errors import DuplicateKeyError

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import server
from safety import (
    APPROVED_REDFLAG_PHRASES,
    SafetyStore,
    detect_safety_risk,
    safety_response,
    validate_redflag_configuration,
)


class MongoCollectionStub:
    def __init__(self):
        self.documents = []

    def create_index(self, *_args, **_kwargs):
        return None

    @staticmethod
    def matches(document, query):
        return all(document.get(key) == value for key, value in query.items())

    def find_one(self, query):
        return next((doc.copy() for doc in self.documents if self.matches(doc, query)), None)

    def insert_one(self, document):
        if any(doc.get("_id") == document.get("_id") for doc in self.documents):
            raise DuplicateKeyError("duplicate")
        if any(
            doc.get("student_id") == document.get("student_id")
            and doc.get("request_id") == document.get("request_id")
            for doc in self.documents
        ):
            raise DuplicateKeyError("duplicate")
        self.documents.append(document.copy())

    def update_one(self, query, update, upsert=False, **_kwargs):
        document = next((doc for doc in self.documents if self.matches(doc, query)), None)
        if document is None and upsert:
            document = {key: value for key, value in query.items()}
            self.documents.append(document)
        if document is None:
            return None
        for key, value in update.get("$inc", {}).items():
            document[key] = document.get(key, 0) + value
        document.update(update.get("$set", {}))
        for key, value in update.get("$setOnInsert", {}).items():
            document.setdefault(key, value)

    def count_documents(self, query):
        return sum(self.matches(document, query) for document in self.documents)

    def find_one_and_update(self, query, update, upsert=False, **_kwargs):
        document = next((doc for doc in self.documents if self.matches(doc, query)), None)
        self.update_one(query, update, upsert=upsert)
        return document.copy() if document is not None else self.find_one(query)


class MongoDatabaseStub:
    def __init__(self):
        self.collections = {}

    def __getitem__(self, name):
        return self.collections.setdefault(name, MongoCollectionStub())


class FailingStatusCollection(MongoCollectionStub):
    fail_next_counter_update = True

    def update_one(self, query, update, upsert=False, **kwargs):
        if self.fail_next_counter_update and "$inc" in update:
            self.fail_next_counter_update = False
            from pymongo.errors import PyMongoError
            raise PyMongoError("simulated counter failure")
        return super().update_one(query, update, upsert=upsert, **kwargs)


class SafetyStoreStub:
    def __init__(self):
        self.count = 0
        self.active = False
        self.screening_required = False
        self.events = {}

    def save_screening_result(self, student_id, result):
        self.active = False
        self.screening_required = result["tier"] == "red"

    def get_status(self, student_id):
        return {
            "student_id": student_id,
            "qualifying_event_count": self.count,
            "restriction_active": self.active,
            "screening_required": self.screening_required,
        }

    def record_event(self, student_id, message, risk, request_id):
        if request_id in self.events:
            return self.events[request_id], self.get_status(student_id), False
        self.count += 1
        self.active = self.count >= 3
        event = {
            "student_id": student_id,
            "message": message,
            "risk_level": risk["risk_level"],
            "immediate": risk["immediate"],
            "qualifying_event_count": self.count,
        }
        self.events[request_id] = event
        return event, self.get_status(student_id), True


@pytest.mark.parametrize(
    "message, qualifies, immediate",
    [
        ("I am stressed about exams.", False, False),
        ("My friend mentioned suicide.", False, False),
        ("I want to kill myself.", True, False),
        ("I have suicidal thoughts right now.", True, True),
    ],
)
def test_contextual_safety_detection(message, qualifies, immediate):
    result = detect_safety_risk(message)
    assert result["qualifies"] is qualifies
    assert result["immediate"] is immediate


@pytest.mark.parametrize("phrase", APPROVED_REDFLAG_PHRASES)
def test_every_approved_phrase_is_detected(phrase):
    result = detect_safety_risk(phrase)
    assert result["qualifies"] is True
    assert result["matched_phrase"] == phrase


@pytest.mark.parametrize(
    "message",
    [
        "I WANT TO DIE!!!",
        "I wish I were dead.",
        "I don't want to live...",
        "I can't go on with this.",
        "I feel completely hopeless.",
        "I took an overdose right now",
    ],
)
def test_case_punctuation_and_apostrophe_variations(message):
    assert detect_safety_risk(message)["qualifies"] is True


def test_quoted_first_person_disclosure_is_detected():
    result = detect_safety_risk("I told my counsellor, 'I want to die.'")
    assert result["qualifies"] is True


@pytest.mark.parametrize(
    "message",
    [
        "I do not want to die.",
        "My friend wants to die.",
        "He said 'I want to die'.",
        "If I want to die, that would be hypothetical.",
        "The movie character said I want to die.",
        "I want to disappear forever from this meeting.",
    ],
)
def test_negated_third_person_hypothetical_and_metaphorical_context_is_rejected(message):
    assert detect_safety_risk(message)["qualifies"] is False


def test_typo_variation_is_detected():
    assert detect_safety_risk("I feel completly hopeless")["qualifies"] is True


def test_configured_phrase_list_replaces_defaults(monkeypatch):
    monkeypatch.setenv("REDFLAG_PHRASES_JSON", '["approved custom phrase"]')
    assert detect_safety_risk("approved custom phrase")["qualifies"] is True
    assert detect_safety_risk("I want to die")["qualifies"] is True


def test_invalid_phrase_configuration_fails_explicitly(monkeypatch):
    monkeypatch.setenv("REDFLAG_PHRASES_JSON", "{invalid")
    with pytest.raises(RuntimeError, match="valid JSON"):
        validate_redflag_configuration()


def test_common_contraction_variations_are_detected():
    assert detect_safety_risk("I do not want to live.")["qualifies"] is True
    assert detect_safety_risk("Life is not worth living.")["qualifies"] is True


def test_historical_disclosure_is_flagged_and_marked_historical():
    result = detect_safety_risk("I attempted suicide five years ago.")
    assert result["qualifies"] is True
    assert result["historical"] is True


def _call_learning(
    store,
    question,
    request_id="request-1",
    answer_function=None,
    learning_store=None,
):
    original_store = server.safety_store
    original_learning_store = server.learning_store
    original_answer = server.answer_student_question
    server.safety_store = store
    server.learning_store = learning_store or Mock()
    server.learning_store.get_all_progress.return_value = []
    server.learning_store.save_interaction.return_value = "interaction-1"
    server.answer_student_question = answer_function or (lambda *_args, **_kwargs: {
        "learning_context": {"message": "normal learning response"},
        "answer": "normal learning response",
        "steps": [],
        "suggested_questions": [],
    })
    try:
        payload = {"question": question, "answer_style": "short"}
        if request_id is not None:
            payload["request_id"] = request_id
        with server.app.test_request_context(
            "/api/learning-response",
            method="POST",
            json=payload,
        ):
            server.g.student_id = "student-1"
            return server.learning_response.__wrapped__()
    finally:
        server.safety_store = original_store
        server.learning_store = original_learning_store
        server.answer_student_question = original_answer


def test_normal_message_reaches_learning_pipeline():
    response, status = _call_learning(SafetyStoreStub(), "How can I practise empathy?")
    assert status == 200
    assert response.get_json()["answer"] == "normal learning response"


def test_flagged_message_never_calls_seek_or_saves_progress():
    store = SafetyStoreStub()
    answer = Mock()
    learning_store = Mock()
    response, status = _call_learning(
        store,
        "I want to die.",
        "safety-request",
        answer_function=answer,
        learning_store=learning_store,
    )
    assert status == 200
    assert response.get_json()["safety"] is True
    answer.assert_not_called()
    learning_store.get_all_progress.assert_not_called()
    learning_store.save_interaction.assert_not_called()
    learning_store.record_learning_event.assert_not_called()
    assert response.get_json().get("suggested_questions") is None


def test_first_second_and_third_events_restrict_without_gemini():
    store = SafetyStoreStub()
    for expected_count in (1, 2, 3):
        response, status = _call_learning(
            store,
            "I want to kill myself.",
            request_id=f"request-{expected_count}",
        )
        assert status == 200
        payload = response.get_json()
        assert payload["safety"] is True
        assert payload["qualifying_event_count"] == expected_count
    assert store.active is True


def test_duplicate_request_does_not_increment_count():
    store = SafetyStoreStub()
    first, _ = _call_learning(store, "I want to kill myself.", "same-request")
    second, _ = _call_learning(store, "I want to kill myself.", "same-request")
    assert first.get_json()["qualifying_event_count"] == 1
    assert second.get_json()["qualifying_event_count"] == 1


def test_missing_request_ids_allow_separate_identical_submissions():
    store = SafetyStoreStub()
    first, _ = _call_learning(store, "I want to die.", request_id=None)
    second, _ = _call_learning(store, "I want to die.", request_id=None)
    assert first.get_json()["qualifying_event_count"] == 1
    assert second.get_json()["qualifying_event_count"] == 2


def test_active_restriction_rejects_chat_before_learning_pipeline():
    store = SafetyStoreStub()
    store.count = 3
    store.active = True
    response, status = _call_learning(store, "What is empathy?", "normal-request")
    assert status == 423
    assert response.get_json()["restriction_active"] is True


def test_immediate_event_has_urgent_guidance():
    event = {
        "risk_level": "immediate",
        "immediate": True,
        "qualifying_event_count": 1,
    }
    payload = safety_response(event, {"qualifying_event_count": 1, "restriction_active": False})
    assert "immediate safety" in payload["message"]
    assert payload["urgent_guidance"]


def _screening_answers(red=False):
    return {
        "phq9": [0] * 8 + ([1] if red else [0]),
        "functional": 0 if red else None,
        "gad7": [0] * 7,
        "asq": [True, False, False, False] if red else [False] * 4,
        "asqNow": True if red else None,
        "context": {},
    }


def test_screening_pass_restores_access_and_red_result_keeps_screening_required():
    original_store = server.safety_store
    original_assessment = server.assessment_store
    store = SafetyStoreStub()
    store.count = 3
    store.screening_required = True
    assessment = Mock()
    server.safety_store = store
    server.assessment_store = assessment
    try:
        with server.app.test_request_context(
            "/api/safety/screening", method="POST", json=_screening_answers()
        ):
            server.g.student_id = "student-1"
            response, status = server.submit_safety_screening.__wrapped__()
        assert status == 200
        assert response.get_json()["access_restored"] is True
        assert store.screening_required is False

        store.screening_required = True
        with server.app.test_request_context(
            "/api/safety/screening", method="POST", json=_screening_answers(red=True)
        ):
            server.g.student_id = "student-1"
            response, status = server.submit_safety_screening.__wrapped__()
        assert status == 200
        assert response.get_json()["access_restored"] is False
        assert store.screening_required is True
    finally:
        server.safety_store = original_store
        server.assessment_store = original_assessment


def test_safety_state_persists_across_store_reinitialization_and_expires():
    database = MongoDatabaseStub()
    first_store = SafetyStore(database)
    risk = detect_safety_risk("I want to kill myself.")
    for index in range(3):
        first_store.record_event("student-1", "I want to kill myself.", risk, f"request-{index}")

    restarted_store = SafetyStore(database)
    status = restarted_store.get_status("student-1")
    assert status["qualifying_event_count"] == 3
    assert status["restriction_active"] is True

    database["student_safety_status"].documents[0]["restriction_expires_at"] = (
        datetime.now(timezone.utc) - timedelta(minutes=1)
    )
    expired = restarted_store.get_status("student-1")
    assert expired["restriction_active"] is False
    assert expired["screening_required"] is True
    restarted_store.save_screening_result("student-1", {"tier": "green"})
    restored = restarted_store.get_status("student-1")
    assert restored["restriction_active"] is False
    assert restored["screening_required"] is False


def test_counter_failure_reconciles_persisted_event():
    database = MongoDatabaseStub()
    database.collections["student_safety_status"] = FailingStatusCollection()
    store = SafetyStore(database)
    risk = detect_safety_risk("I want to die.")

    with pytest.raises(Exception, match="simulated counter failure"):
        store.record_event("student-1", "I want to die.", risk, "failure-request")

    assert database["safety_events"].count_documents({"student_id": "student-1"}) == 1
    status = store.get_status("student-1")
    assert status["qualifying_event_count"] == 1


def test_mongodb_duplicate_request_returns_existing_event_without_incrementing():
    store = SafetyStore(MongoDatabaseStub())
    risk = detect_safety_risk("I want to die.")
    first, first_status, first_created = store.record_event(
        "student-1", "I want to die.", risk, "same-request"
    )
    second, second_status, second_created = store.record_event(
        "student-1", "I want to die.", risk, "same-request"
    )
    assert first_created is True
    assert second_created is False
    assert first["_id"] == second["_id"]
    assert first_status["qualifying_event_count"] == second_status["qualifying_event_count"] == 1

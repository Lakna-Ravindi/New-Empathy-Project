 """Safety detection, persistence, and chatbot access control."""

import hashlib
import json
import os
import re
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from uuid import uuid4

from pymongo import ASCENDING, ReturnDocument
from pymongo.errors import DuplicateKeyError, OperationFailure, PyMongoError


RESTRICTION_DURATION = timedelta(hours=24)

APPROVED_REDFLAG_PHRASES = (
    "want to die",
    "wish i were dead",
    "don't want to live",
    "life isn't worth living",
    "no reason to live",
    "suicidal thoughts",
    "attempted suicide",
    "want to kill myself",
    "feel like killing myself",
    "ending my life",
    "harming myself",
    "hurting myself",
    "cut myself",
    "took an overdose",
    "i want to disappear forever",
    "i can't go on like this",
    "i can't go on with this",
    "feeling trapped",
    "i feel completely hopeless",
    "there is no way out",
)

_FIRST_PERSON = re.compile(
    r"\b(i|im|i'm|myself|me)\b", re.IGNORECASE
)
_THIRD_PERSON = re.compile(
    r"\b(my friend|my brother|my sister|my parent|my partner|"
    r"he|she|they|them|his|her|their|friend|someone|another person|a friend)\b",
    re.IGNORECASE,
)
_CONTEXT_DISQUALIFIERS = re.compile(
    r"\b(imagine|imagining|suppose|hypothetical|hypothetically|"
    r"lyrics|song|movie|fiction|story|character|class discussion)\b",
    re.IGNORECASE,
)
_METAPHOR_CONTEXT = re.compile(
    r"\b(from this (meeting|class|conversation|chat|job|project)|"
    r"from social media|from work|from school|for a while|"
    r"delete my account|leave this place)\b",
    re.IGNORECASE,
)
_NEGATED_RED_FLAG = re.compile(
    r"\b(do not|dont|never|not)\s+(want to die|wish i were dead|"
    r"end my life|kill myself|harm myself|hurt myself|cut myself|"
    r"take an overdose|attempt suicide)\b",
    re.IGNORECASE,
)
_IMMEDIATE = re.compile(
    r"\b(right now|now|currently|tonight|today|this moment|"
    r"have a plan|made a plan|about to|going to do it|already did|"
    r"just took|already took|took an overdose)\b",
    re.IGNORECASE,
)


def _normalize_text(message):
    text = str(message or "").lower()
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"\bdo not\b", "dont", text)
    text = re.sub(r"\bis not\b", "isnt", text)
    text = re.sub(r"\bcan not\b|\bcannot\b", "cant", text)
    text = re.sub(r"\bi am\b", "im", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("'", "")


def _configured_phrases():
    raw = os.getenv("REDFLAG_PHRASES_JSON", "").strip()
    if not raw:
        return APPROVED_REDFLAG_PHRASES
    try:
        configured = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError("REDFLAG_PHRASES_JSON must contain valid JSON.") from error
    if (
        not isinstance(configured, list)
        or not configured
        or any(not isinstance(phrase, str) or not phrase.strip() for phrase in configured)
    ):
        raise RuntimeError("REDFLAG_PHRASES_JSON must be a non-empty JSON list of phrases.")
    return tuple(dict.fromkeys((*APPROVED_REDFLAG_PHRASES, *(
        phrase.strip() for phrase in configured
    ))))


def validate_redflag_configuration():
    """Validate safety phrase configuration during application startup."""
    _configured_phrases()


def _fuzzy_phrase_match(text, phrase):
    phrase_tokens = _normalize_text(phrase).split()
    text_tokens = text.split()
    width = len(phrase_tokens)
    if not width or width > len(text_tokens):
        return False
    for start in range(len(text_tokens) - width + 1):
        window = text_tokens[start:start + width]
        if all(
            token == expected
            or (
                len(expected) >= 5
                and len(token) >= 5
                and token[0] == expected[0]
                and SequenceMatcher(None, token, expected).ratio() >= 0.88
            )
            for token, expected in zip(window, phrase_tokens)
        ):
            return True
    return False


def _has_disqualifying_context(text):
    if _CONTEXT_DISQUALIFIERS.search(text):
        return True
    if _METAPHOR_CONTEXT.search(text):
        return True
    if _THIRD_PERSON.search(text) and not _FIRST_PERSON.search(text):
        return True
    return False


def _sentences(text):
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _is_hypothetical_sentence(sentence):
    return bool(re.match(
        r"^(if|suppose|imagine|hypothetically)\b",
        sentence,
        re.IGNORECASE,
    )) and not bool(re.search(r"\b(i don't know|i dont know)\s+if\b", sentence))


def _is_unrelated_quoted_disclosure(message, phrase):
    phrase_text = _normalize_text(phrase)
    for match in re.finditer(r"""(["“'])(.*?)\1""", str(message or ""), re.DOTALL):
        if phrase_text not in _normalize_text(match.group(2)):
            continue
        outside = f"{message[:match.start()]} {message[match.end():]}"
        if not re.search(r"\b(i|im)\b", _normalize_text(outside)):
            return True
    return False


def _is_third_person_attribution(sentence, phrase):
    third_match = list(_THIRD_PERSON.finditer(sentence))
    if not third_match:
        return False
    last_third = third_match[-1]
    return not bool(_FIRST_PERSON.search(sentence[last_third.end():]))


def detect_safety_risk(message):
    """Return a structured safety result without affecting SEEK classification."""
    text = _normalize_text(message)
    matched_phrase = None
    matched_sentence = None
    for sentence in _sentences(text) or [text]:
        for phrase in _configured_phrases():
            if _normalize_text(phrase) in sentence or _fuzzy_phrase_match(sentence, phrase):
                matched_phrase = phrase
                matched_sentence = sentence
                break
        if matched_phrase:
            break

    qualifies = bool(matched_phrase)
    if qualifies and _is_third_person_attribution(matched_sentence, matched_phrase):
        qualifies = False
    if qualifies and (
        _has_disqualifying_context(matched_sentence)
        or _is_hypothetical_sentence(matched_sentence)
        or _NEGATED_RED_FLAG.search(matched_sentence)
        or _is_unrelated_quoted_disclosure(message, matched_phrase)
    ):
        qualifies = False

    immediate = qualifies and bool(_IMMEDIATE.search(matched_sentence))
    historical = bool(
        qualifies and re.search(
            r"\b(years ago|last year|months ago|previously|when i was|in the past|"
            r"once|tried before|attempted)\b",
            matched_sentence,
            re.IGNORECASE,
        )
    )
    if not qualifies:
        return {
            "qualifies": False,
            "risk_level": "none",
            "immediate": False,
            "historical": False,
            "category": None,
        }

    return {
        "qualifies": True,
        "risk_level": "immediate" if immediate else "high",
        "immediate": immediate,
        "historical": historical,
        "category": (
            "suicidal_ideation"
            if any(term in _normalize_text(matched_phrase) for term in ("die", "dead", "live", "suicid", "life"))
            else "self_harm"
        ),
        "matched_phrase": matched_phrase,
    }


def _utc_now():
    return datetime.now(timezone.utc)


def _transactions_unavailable(error):
    return (
        error.code in {20, 263, 303}
        or "transaction" in str(error).lower()
        and "support" in str(error).lower()
    )


def _parse_contacts():
    raw = os.getenv("COUNSELLOR_CONTACTS_JSON", "").strip()
    if not raw:
        return []
    try:
        contacts = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError("COUNSELLOR_CONTACTS_JSON must contain valid JSON.") from error
    if not isinstance(contacts, list):
        raise RuntimeError("COUNSELLOR_CONTACTS_JSON must be a JSON list.")
    return [
        item for item in contacts
        if isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and isinstance(item.get("contact"), str)
    ]


def support_payload():
    return {
        "counsellor_contacts": _parse_contacts(),
        "urgent_guidance": (
            "If you may act on these thoughts or are in immediate danger, "
            "call local emergency services or go to the nearest emergency department. "
            "Please stay with someone you trust and move away from anything you could use to hurt yourself."
        ),
    }


class SafetyStore:
    """Mongo-backed safety state using the application's existing database."""

    def __init__(self, db, client=None):
        if db is None:
            raise PyMongoError("MongoDB safety database is unavailable.")
        self.events = db["safety_events"]
        self.status = db["student_safety_status"]
        self.client = client
        self.events.create_index(
            [("student_id", ASCENDING), ("request_id", ASCENDING)],
            unique=True,
            name="uq_safety_event_request",
        )
        self.events.create_index([("student_id", ASCENDING), ("created_at", ASCENDING)])
        self.status.create_index(
            [("student_id", ASCENDING)],
            unique=True,
            name="uq_student_safety_status",
        )

    def get_status(self, student_id):
        current = self.status.find_one({"student_id": student_id}) or {
            "student_id": student_id,
            "qualifying_event_count": 0,
            "restriction_active": False,
            "screening_required": False,
        }
        expiry = current.get("restriction_expires_at")
        if expiry and expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        if current.get("restriction_active") and expiry and expiry <= _utc_now():
            current = self.status.find_one_and_update(
                {"student_id": student_id, "restriction_active": True},
                {
                    "$set": {
                        "restriction_active": False,
                        "screening_required": True,
                        "updated_at": _utc_now(),
                    }
                },
                return_document=ReturnDocument.AFTER,
            ) or current
        if hasattr(self.events, "count_documents"):
            event_count = self.events.count_documents({"student_id": student_id})
            if event_count != current.get("qualifying_event_count", 0):
                current = self.status.find_one_and_update(
                    {"student_id": student_id},
                    {
                        "$set": {
                            "qualifying_event_count": event_count,
                            "updated_at": _utc_now(),
                        }
                    },
                    upsert=True,
                    return_document=ReturnDocument.AFTER,
                ) or current
        return current

    def record_event(self, student_id, message, risk, request_id):
        request_id = request_id or str(uuid4())
        event_id = hashlib.sha256(
            f"{student_id}:{request_id}".encode("utf-8")
        ).hexdigest()
        event = {
            "_id": event_id,
            "student_id": student_id,
            "request_id": request_id,
            "message": message,
            "risk_level": risk["risk_level"],
            "category": risk["category"],
            "immediate": risk["immediate"],
            "historical": risk.get("historical", False),
            "created_at": _utc_now(),
        }
        def persist(session=None):
            options = {"session": session} if session is not None else {}
            self.events.insert_one(event, **options)
            now = _utc_now()
            status = self.status.find_one_and_update(
                {"student_id": student_id},
                {
                    "$inc": {"qualifying_event_count": 1},
                    "$set": {"updated_at": now},
                    "$setOnInsert": {
                        "student_id": student_id,
                        "restriction_active": False,
                        "screening_required": False,
                    },
                },
                upsert=True,
                return_document=ReturnDocument.AFTER,
                **options,
            )
            return status, now

        try:
            if self.client is not None:
                with self.client.start_session() as session:
                    with session.start_transaction():
                        status, now = persist(session)
            else:
                status, now = persist()
        except DuplicateKeyError:
            existing = self.events.find_one(
                {"student_id": student_id, "request_id": request_id}
            )
            if existing is None:
                raise
            return existing, self.get_status(student_id), False
        except OperationFailure as error:
            if not _transactions_unavailable(error):
                self._reconcile_status(student_id)
                raise
            try:
                status, now = persist()
            except PyMongoError:
                self._reconcile_status(student_id)
                raise
        except PyMongoError:
            self._reconcile_status(student_id)
            raise

        count = status["qualifying_event_count"]
        event_update = {"qualifying_event_count": count}
        if count >= 3 and not status.get("restriction_active"):
            restriction_start = now
            restriction_expiry = now + RESTRICTION_DURATION
            status = self.status.find_one_and_update(
                {"student_id": student_id},
                {
                    "$set": {
                        "restriction_active": True,
                        "screening_required": False,
                        "restriction_started_at": restriction_start,
                        "restriction_expires_at": restriction_expiry,
                        "updated_at": now,
                    }
                },
                return_document=ReturnDocument.AFTER,
            )
        event_update["restriction_active"] = bool(status.get("restriction_active"))
        self.events.update_one({"_id": event_id}, {"$set": event_update})
        return {**event, **event_update}, status, True

    def _reconcile_status(self, student_id):
        if not hasattr(self.events, "count_documents"):
            return
        count = self.events.count_documents({"student_id": student_id})
        self.status.update_one(
            {"student_id": student_id},
            {
                "$set": {
                    "qualifying_event_count": count,
                    "updated_at": _utc_now(),
                },
                "$setOnInsert": {
                    "student_id": student_id,
                    "restriction_active": False,
                    "screening_required": False,
                },
            },
            upsert=True,
        )
        current = self.status.find_one({"student_id": student_id}) or {}
        if (
            count >= 3
            and not current.get("restriction_active")
            and not current.get("screening_required")
            and not current.get("last_screening_at")
        ):
            now = _utc_now()
            self.status.update_one(
                {"student_id": student_id},
                {
                    "$set": {
                        "restriction_active": True,
                        "restriction_started_at": now,
                        "restriction_expires_at": now + RESTRICTION_DURATION,
                        "updated_at": now,
                    }
                },
            )

    def save_screening_result(self, student_id, result):
        now = _utc_now()
        self.status.update_one(
            {"student_id": student_id},
            {
                "$set": {
                    "screening_required": result["tier"] == "red",
                    "restriction_active": False,
                    "last_screening_tier": result["tier"],
                    "last_screening_at": now,
                    "updated_at": now,
                }
            },
            upsert=True,
        )


def safety_response(event, status):
    count = status.get("qualifying_event_count", event.get("qualifying_event_count", 1))
    restricted = bool(status.get("restriction_active"))
    payload = {
        "safety": True,
        "risk_level": event.get("risk_level", "high"),
        "qualifying_event_count": count,
        "restriction_active": restricted,
        "screening_required": bool(status.get("screening_required")),
        "message": (
            "Thank you for telling me. You deserve immediate, real-world support. "
            "Please contact someone you trust and a qualified counsellor now."
        ),
        **support_payload(),
    }
    if event.get("immediate"):
        payload["message"] = (
            "I’m really sorry you’re facing this. Your immediate safety matters more than continuing this chat. "
            "Please call emergency services or go to the nearest emergency department now, and stay with someone you trust."
        )
    if restricted:
        payload.update({
            "message": (
                "For your safety, chatbot access is temporarily paused for 24 hours. "
                "Please contact a qualified counsellor or emergency service now."
            ),
            "restriction_expires_at": status.get("restriction_expires_at"),
        })
    return payload

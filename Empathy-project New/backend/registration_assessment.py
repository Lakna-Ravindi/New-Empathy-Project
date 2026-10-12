# backend/registration_assessment.py
"""
Scoring + storage for the two tests taken during registration:
  1. Mental-health screening (PHQ-9, GAD-7, ASQ + optional context)
  2. 20-question empathy pre-skill assessment

Scores are stored in their own collections, keyed by student_id, and are
never returned to the frontend.
"""
import os
from datetime import datetime, timezone

from pymongo import ASCENDING, MongoClient
from pymongo.errors import PyMongoError

# ---------------------------------------------------------------
# Screening
# ---------------------------------------------------------------
FREQ_VALUES = {0, 1, 2, 3}

CONTEXT_ALLOWED = {
    "in_care": {"yes", "no"},
    "recent_crisis": {"yes", "no", "prefer_not"},
    "substance_coping": {"never", "sometimes", "often"},
    "feels_safe": {"yes", "no", "prefer_not"},
}

# TODO: align with your own clinical protocol / ethics approval.
RED_PHQ9_MIN = 20
RED_GAD7_MIN = 15
AMBER_PHQ9_MIN = 10
AMBER_GAD7_MIN = 10


def _int_list(value, length, label):
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(isinstance(v, bool) or not isinstance(v, int) or v not in FREQ_VALUES for v in value)
    ):
        raise ValueError(f"Invalid {label} answers.")
    return value


def score_screening(raw):
    if not isinstance(raw, dict):
        raise ValueError("Screening answers are required.")

    phq9 = _int_list(raw.get("phq9"), 9, "PHQ-9")
    gad7 = _int_list(raw.get("gad7"), 7, "GAD-7")
    phq9_total = sum(phq9)
    gad7_total = sum(gad7)

    functional = raw.get("functional")
    if phq9_total > 0:
        if isinstance(functional, bool) or not isinstance(functional, int) or functional not in FREQ_VALUES:
            raise ValueError("Invalid functional-difficulty answer.")
    else:
        functional = None

    asq = raw.get("asq")
    if not isinstance(asq, list) or len(asq) != 4 or any(not isinstance(v, bool) for v in asq):
        raise ValueError("Invalid safety-question answers.")
    any_asq_yes = any(asq)

    asq_now = raw.get("asqNow")
    if any_asq_yes:
        if not isinstance(asq_now, bool):
            raise ValueError("Invalid safety-question answers.")
    else:
        asq_now = None

    acute = any_asq_yes and asq_now is True

    context = {}
    raw_context = raw.get("context") or {}
    if isinstance(raw_context, dict):
        for key, allowed in CONTEXT_ALLOWED.items():
            value = raw_context.get(key)
            if isinstance(value, str) and value in allowed:
                context[key] = value

    if acute or any_asq_yes or phq9_total >= RED_PHQ9_MIN or gad7_total >= RED_GAD7_MIN:
        tier = "red"
    elif phq9_total >= AMBER_PHQ9_MIN or gad7_total >= AMBER_GAD7_MIN or phq9[8] > 0:
        tier = "amber"
    else:
        tier = "green"

    return {
        "phq9_items": phq9,
        "phq9_total": phq9_total,
        "phq9_item9": phq9[8],
        "functional_difficulty": functional,
        "gad7_items": gad7,
        "gad7_total": gad7_total,
        "asq_items": asq,
        "asq_positive": any_asq_yes,
        "asq_acute_now": acute,
        "context": context,
        "acute": acute,
        "tier": tier,
    }


# ---------------------------------------------------------------
# Pre-skill assessment (answer key stays on the server)
# question id -> (skill_id, correct option)
# skill_01..skill_08 follow the order of the 8 SEEK skills; change them if your
# content_skills.json uses different ids.
# ---------------------------------------------------------------
PRE_SKILL_KEY = {
    "q01": ("skill_01", "c"), "q02": ("skill_01", "a"), "q03": ("skill_01", "d"),
    "q04": ("skill_02", "b"), "q05": ("skill_02", "c"), "q06": ("skill_02", "d"),
    "q07": ("skill_03", "b"), "q08": ("skill_03", "c"), "q09": ("skill_03", "a"),
    "q10": ("skill_04", "c"), "q11": ("skill_04", "d"), "q12": ("skill_04", "a"),
    "q13": ("skill_05", "b"), "q14": ("skill_05", "d"),
    "q15": ("skill_06", "c"), "q16": ("skill_06", "d"),
    "q17": ("skill_07", "c"), "q18": ("skill_07", "b"),
    "q19": ("skill_08", "a"), "q20": ("skill_08", "d"),
}
VALID_OPTIONS = {"a", "b", "c", "d"}


def score_pre_skill(raw):
    if not isinstance(raw, dict):
        raise ValueError("Pre-skill assessment answers are required.")

    missing = [qid for qid in PRE_SKILL_KEY if qid not in raw]
    if missing:
        raise ValueError("Please answer all 20 questions.")
    if any(not isinstance(raw[qid], str) or raw[qid] not in VALID_OPTIONS for qid in PRE_SKILL_KEY):
        raise ValueError("Invalid pre-skill assessment answers.")

    skill_scores = {}
    correct_total = 0
    for qid, (skill_id, correct) in PRE_SKILL_KEY.items():
        entry = skill_scores.setdefault(skill_id, {"correct": 0, "total": 0})
        entry["total"] += 1
        if raw[qid] == correct:
            entry["correct"] += 1
            correct_total += 1

    total = len(PRE_SKILL_KEY)
    return {
        "answers": {qid: raw[qid] for qid in PRE_SKILL_KEY},
        "score": correct_total,
        "total": total,
        "percentage": round(correct_total / total * 100, 1),
        "skill_scores": skill_scores,
    }


# ---------------------------------------------------------------
# Storage
# ---------------------------------------------------------------
class RegistrationAssessmentStore:
    def __init__(self, db=None):
        # server.py passes student_store.db so everything lives in the same
        # database as the users collection. The fallback below uses the same
        # variable names as auth.py.
        if db is None:
            client = MongoClient(
                os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
                serverSelectionTimeoutMS=5000,
            )
            db = client[os.getenv("MONGODB_DATABASE", "empathy_learning")]

        self.screening = db["screening_results"]
        self.pre_skill = db["pre_skill_assessments"]
        try:
            self.screening.create_index([("student_id", ASCENDING)], unique=True)
            self.pre_skill.create_index([("student_id", ASCENDING)], unique=True)
        except PyMongoError:
            pass  # indexes are created on the next start if Mongo was unreachable

    @staticmethod
    def _now():
        return datetime.now(timezone.utc)

    def save_screening(self, student_id, result):
        self.screening.insert_one({"student_id": student_id, **result, "created_at": self._now()})

    def save_pre_skill(self, student_id, result):
        self.pre_skill.insert_one({"student_id": student_id, **result, "created_at": self._now()})

    def delete_for_student(self, student_id):
        self.screening.delete_many({"student_id": student_id})
        self.pre_skill.delete_many({"student_id": student_id})
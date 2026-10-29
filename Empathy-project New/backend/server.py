# backend/server.py
import json
import os
import re
from pathlib import Path

import bcrypt
from flask import Flask, g, jsonify, request
from bson.errors import InvalidId
from pymongo.errors import DuplicateKeyError, PyMongoError
from dotenv import load_dotenv

from auth import VALID_AGE_GROUPS, VALID_GENDERS, create_access_token, get_user_store, require_auth, require_role
from general_empathy_service import EmpathyServiceError, generate_empathy_response
from learning.pedagogical_controller import answer_student_question, belongs_to_skill
from learning.interaction_store import LearningStore
from learning.progress_service import ProgressService
from registration_assessment import (
    RegistrationAssessmentStore,
    score_pre_skill,
    score_screening,
)

from flask_cors import CORS


def is_greeting(text: str) -> bool:
    normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    normalized = " ".join(normalized.split())

    greetings = {
        "hi",
        "hii",
        "hiii",
        "hello",
        "helloo",
        "hey",
        "heyy",
        "hiya",
        "good morning",
        "good afternoon",
        "good evening",
    }

    return normalized in greetings


app = Flask(__name__)

CORS(
    app,
    resources={r"/api/*": {"origins": "http://localhost:5173"}},
    allow_headers=["Content-Type", "Authorization"],
)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:5173"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / "backend" / ".env")

with open(BASE_DIR / "output" / "knowledge_base.json", "r", encoding="utf-8") as file:
    knowledge_base = json.load(file)

with open(BASE_DIR / "output" / "highlighted_keywords.json", "r", encoding="utf-8") as file:
    highlighted_keywords = json.load(file)

with open(BASE_DIR / "output" / "keyword_skill_map.json", "r", encoding="utf-8") as file:
    keyword_skill_map = json.load(file)


def _item_with_id(item, item_id, title_key="title"):
    """Add a stable ID without changing the source content shape."""
    if isinstance(item, dict):
        return {**item, "id": item_id}
    return {"id": item_id, title_key: item, "content": item}


def _normalize_skill(skill, index):
    skill_id = skill.get("skill_id") or f"skill_{index:02d}"
    skill_number = skill_id.rsplit("_", 1)[-1]

    objectives = []
    for item_index, item in enumerate(
        skill.get("objectives", skill.get("learning_objectives", [])), 1
    ):
        authored_id = item.get("objective_id") if isinstance(item, dict) else None
        objectives.append(
            _item_with_id(
                item,
                authored_id or f"objective_{skill_number}_{item_index:02d}",
                "title",
            )
        )
    activities = [
        _item_with_id(item, f"activity_{skill_number}_{item_index:02d}")
        for item_index, item in enumerate(skill.get("activities", []), 1)
    ]
    videos = [
        _item_with_id(item, f"video_{skill_number}_{item_index:02d}")
        for item_index, item in enumerate(skill.get("videos", []), 1)
    ]
    quizzes = [
        _item_with_id(item, f"quiz_{skill_number}_{item_index:02d}")
        for item_index, item in enumerate(
            skill.get("quizzes", skill.get("quiz", [])), 1
        )
    ]

    for item in objectives:
        item["objective_id"] = item.get("objective_id") or item["id"]
    for item in activities:
        item["activity_id"] = item["id"]
    for item in videos:
        item["video_id"] = item["id"]
    for item in quizzes:
        item["quiz_id"] = item["id"]

    return {
        **skill,
        "skill_id": skill_id,
        "title": re.sub(
            r"^Skill\s+\d+\s*:\s*",
            "",
            skill.get("title") or skill.get("skill_title", ""),
            flags=re.IGNORECASE,
        ),
        "objectives": objectives,
        "learning_objectives": objectives,
        "activities": activities,
        "videos": videos,
        "quizzes": quizzes,
    }


def load_content_skills():
    configured_path = os.getenv("CONTENT_KB_PATH")
    source_paths = [
        Path(configured_path)
        if configured_path
        else BASE_DIR / "output" / "content_skills.json"
    ]
    if not configured_path:
        source_paths.append(BASE_DIR / "output" / "knowledge_base.json")
        source_paths.append(Path.home() / "Downloads" / "knowledge_base.json")

    for source_path in source_paths:
        if not source_path.exists():
            continue
        with source_path.open("r", encoding="utf-8") as file:
            source = json.load(file)
        if isinstance(source, dict):
            source = source.get("skills", [])
        if isinstance(source, list) and all(isinstance(item, dict) for item in source):
            if not source or any(item.get("skill_id") for item in source):
                return [_normalize_skill(skill, index) for index, skill in enumerate(source, 1)]

    return []


content_skills = load_content_skills()

nodes_by_id = {node["id"]: node for node in knowledge_base}


def attach_progress_item_id(learning_context):
    skill = learning_context.get("skill") or {}
    skill_id = skill.get("id") or skill.get("skill_id")
    skill_id, _ = resolve_legacy_progress_ids(skill_id, "")
    if skill.get("id"):
        skill["id"] = skill_id
    if skill.get("skill_id"):
        skill["skill_id"] = skill_id

    chapter = next(
        (
            item
            for item in progress_service.structure.get("chapters", [])
            if item.get("chapter_id") == skill_id
        ),
        None,
    )
    if chapter is None:
        return

    activity = learning_context.get("recommended_activity")
    objective_context = learning_context.get("learning_objective") or {}
    objective_id = (
        objective_context.get("objective_id")
        or objective_context.get("id")
    )
    if isinstance(activity, dict) and activity.get("id"):
        activity_id = activity.get("id")
    else:
        activity_id = None

    if not activity_id and objective_id:
        objective = next(
            (
                item
                for item in chapter.get("objectives", [])
                if item.get("objective_id") == objective_id
            ),
            None,
        )
        selected_item = next(
            (
                item
                for item in (objective or {}).get("items", [])
                if item.get("type") == "activity"
            ),
            None,
        )
        if selected_item:
            activity = {
                "id": selected_item["item_id"],
                "type": selected_item.get("type"),
                "title": selected_item.get("title"),
                "content": selected_item.get("content"),
                "objective_id": objective_id,
            }
            learning_context["recommended_activity"] = activity
            activity_id = activity["id"]

    if not isinstance(activity, dict) or not activity_id:
        return

    type_aliases = {"practice": "activity", "assessment": "quiz"}
    activity_type = str(activity.get("type") or "").lower()
    activity_type = type_aliases.get(activity_type, activity_type)
    known_item_types = {"activity", "video", "quiz", "reflection", "text"}
    activity_text = " ".join(
        str(activity.get(field) or "")
        for field in ("title", "content")
    ).lower()
    activity_terms = set(re.findall(r"[a-z0-9]+", activity_text))
    candidates = [
        item
        for objective in chapter.get("objectives", [])
        for item in objective.get("items", [])
        if item.get("item_id") and (
            not activity_type
            or activity_type not in known_item_types
            or item.get("type") == activity_type
        )
    ]
    if not candidates:
        return

    activity_node = nodes_by_id.get(activity.get("id"))
    visited = set()
    while activity_node and activity_node.get("id") not in visited:
        visited.add(activity_node.get("id"))
        if activity_node.get("objective_id"):
            break
        activity_node = nodes_by_id.get(activity_node.get("parent_id"))
    objective_id = (
        activity.get("objective_id")
        or (activity_node or {}).get("objective_id")
        or objective_id
    )
    if objective_id:
        objective_candidates = [
            item
            for objective in chapter.get("objectives", [])
            if objective.get("objective_id") == objective_id
            for item in objective.get("items", [])
            if item in candidates
        ]
        if objective_candidates:
            candidates = objective_candidates

    direct_ids = {
        activity.get("id"),
        activity.get("progress_item_id"),
        activity.get("item_id"),
        activity.get("activity_id"),
        activity.get("video_id"),
        activity.get("quiz_id"),
    }
    direct_matches = [
        item for item in candidates if item.get("item_id") in direct_ids
    ]
    if len(direct_matches) == 1:
        selected_item = direct_matches[0]
    else:
        selected_item = None

    def match_score(item):
        item_text = " ".join(
            str(item.get(field) or "")
            for field in ("title", "content")
        ).lower()
        return len(activity_terms & set(re.findall(r"[a-z0-9]+", item_text)))

    if selected_item is None:
        scored_candidates = [
            (match_score(item), item)
            for item in candidates
        ]
        best_score = max(score for score, _ in scored_candidates)
        best_matches = [
            item for score, item in scored_candidates if score == best_score
        ]
        if best_score == 0 or len(best_matches) != 1:
            return
        selected_item = best_matches[0]

    if not selected_item.get("item_id"):
        return
    activity["progress_item_id"] = selected_item["item_id"]
    activity["progress_item_type"] = selected_item.get("type")
    activity["progress_objective_id"] = next(
        (
            objective.get("objective_id")
            for objective in chapter.get("objectives", [])
            if any(
                item.get("item_id") == selected_item["item_id"]
                for item in objective.get("items", [])
            )
        ),
        None,
    )


def resolve_legacy_progress_ids(skill_id, item_id):
    """Translate older knowledge-base node IDs to progress structure IDs."""
    chapter_ids = {
        chapter.get("chapter_id")
        for chapter in progress_service.structure.get("chapters", [])
    }
    if skill_id not in chapter_ids:
        skill_node = nodes_by_id.get(skill_id)
        if skill_node:
            skill_id = skill_node.get("skill_id") or skill_id
            if skill_id not in chapter_ids and skill_node.get("type") == "chapter":
                chapter_nodes = [
                    node
                    for node in knowledge_base
                    if node.get("type") == "chapter"
                ]
                chapter_index = next(
                    (
                        index
                        for index, node in enumerate(chapter_nodes)
                        if node.get("id") == skill_node.get("id")
                    ),
                    None,
                )
                if chapter_index is not None:
                    ordered_chapters = progress_service.structure.get("chapters", [])
                    if chapter_index < len(ordered_chapters):
                        skill_id = ordered_chapters[chapter_index].get("chapter_id")

    if item_id in {
        item.get("item_id")
        for chapter in progress_service.structure.get("chapters", [])
        for objective in chapter.get("objectives", [])
        for item in objective.get("items", [])
    }:
        return skill_id, item_id

    item_node = nodes_by_id.get(item_id)
    if not item_node:
        return skill_id, item_id

    chapter = next(
        (
            chapter
            for chapter in progress_service.structure.get("chapters", [])
            if chapter.get("chapter_id") == skill_id
        ),
        None,
    )
    if chapter is None:
        return skill_id, item_id

    return skill_id, item_id


def _skill_summary(skill):
    return {
        "skill_id": skill["skill_id"],
        "title": skill["title"],
        "description": skill.get("description", ""),
    }


@app.get("/api/content/skills")
def list_content_skills():
    return jsonify({"skills": [_skill_summary(skill) for skill in content_skills]}), 200


@app.get("/api/content/skills/<skill_id>")
def get_content_skill(skill_id):
    skill = next(
        (item for item in content_skills if item["skill_id"] == skill_id),
        None,
    )
    if skill is None:
        return jsonify({"error": "Skill was not found."}), 404
    return jsonify(skill), 200

learning_store = LearningStore()
student_store = get_user_store()
progress_service = ProgressService()

# Screening / pre-skill results live in the SAME database as the users collection.
assessment_store = RegistrationAssessmentStore(db=student_store.db)


def get_skill_objectives(skill_id):
    chapter = next(
        (
            chapter
            for chapter in progress_service.structure.get("chapters", [])
            if chapter.get("chapter_id") == skill_id
        ),
        None,
    )
    if chapter is not None:
        return [
            {
                "id": objective["objective_id"],
                "title": objective["title"],
                "content": objective.get("content", ""),
            }
            for objective in chapter.get("objectives", [])
        ]

    return [
        {
            "id": node["id"],
            "title": node["title"],
            "content": node["content"],
        }
        for node in knowledge_base
        if node.get("type") == "learning_objective"
        and belongs_to_skill(node, skill_id, nodes_by_id)
    ]


def next_uncompleted_objective(skill_id, completed_ids):
    return next(
        (
            objective
            for objective in get_skill_objectives(skill_id)
            if objective["id"] not in completed_ids
        ),
        None,
    )


# ---------------------------------------------------
# Authentication APIs
# ---------------------------------------------------

@app.post("/api/auth/register")
def register_student():
    body = request.get_json(silent=True) or {}

    username = (body.get("username") or "").strip().lower()
    email = (body.get("email") or "").strip().lower()
    gender = (body.get("gender") or "").strip()
    password = body.get("password") or ""

    for field, value in (("Email", email), ("Gender", gender), ("Username", username), ("Password", password)):
        if not value:
            return jsonify({"error": f"{field} is required"}), 400

    if not re.fullmatch(r"[a-z0-9_]{3,30}", username):
        return jsonify({
            "error": "Username must be 3-30 characters and use only letters, numbers, or underscores."
        }), 400

    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        return jsonify({
            "error": "Please provide a valid email address."
        }), 400

    if len(password) < 8:
        return jsonify({
            "error": "Password must contain at least 8 characters."
        }), 400

    if gender not in VALID_GENDERS:
        return jsonify({"error": "Invalid gender"}), 400
    try:
        student = student_store.create_student(username, email, password, gender)

        # Login token is returned immediately after registration.
        stored_student = student_store.find_by_username(username)
        token = create_access_token(stored_student)

        response_user = {
            **student,
            "role": "student",
        }
        return jsonify({
            "message": "Registration successful.",
            "user": response_user,
            "student": response_user,
            "access_token": token,
        }), 201

    except DuplicateKeyError:
        duplicate = "Email already exists" if student_store.find_by_email(email) else "Username already exists"
        return jsonify({"error": duplicate}), 409

    except PyMongoError:
        return jsonify({
            "error": "Could not create the student account."
        }), 503


def _clean_registration(body):
    """Same rules as /api/auth/register. Returns (data, None) or (None, error_response)."""
    username = (body.get("username") or "").strip().lower()
    email = (body.get("email") or "").strip().lower()
    gender = (body.get("gender") or "").strip()
    password = body.get("password") or ""

    for field, value in (("Email", email), ("Gender", gender), ("Username", username), ("Password", password)):
        if not value:
            return None, (jsonify({"error": f"{field} is required"}), 400)

    if not re.fullmatch(r"[a-z0-9_]{3,30}", username):
        return None, (jsonify({
            "error": "Username must be 3-30 characters and use only letters, numbers, or underscores."
        }), 400)

    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        return None, (jsonify({"error": "Please provide a valid email address."}), 400)

    if len(password) < 8:
        return None, (jsonify({"error": "Password must contain at least 8 characters."}), 400)

    if gender not in VALID_GENDERS:
        return None, (jsonify({"error": "Invalid gender"}), 400)

    return {"username": username, "email": email, "gender": gender, "password": password}, None


def _duplicate_error(username, email):
    if student_store.find_by_username(username):
        return jsonify({"error": "Username already exists"}), 409
    if student_store.find_by_email(email):
        return jsonify({"error": "Email already exists"}), 409
    return None


@app.post("/api/auth/register/validate")
def validate_registration():
    """Step 1 of sign-up: check the details, save nothing."""
    body = request.get_json(silent=True) or {}
    data, error = _clean_registration(body)
    if error:
        return error

    try:
        duplicate = _duplicate_error(data["username"], data["email"])
    except PyMongoError:
        return jsonify({"error": "Could not check the registration details."}), 503
    if duplicate:
        return duplicate

    return jsonify({"valid": True}), 200



@app.post("/api/auth/register/complete")
def register_with_assessments():
    """
    Final step of sign-up. Saves the account together with the screening and
    the pre-skill assessment. Scores are stored in separate collections and
    are NOT returned (only the tier, for the closing message).

    Safety net: a red-tier screening is never saved. The frontend already
    stops these students after test 1, this blocks direct API calls.
    """
    body = request.get_json(silent=True) or {}
    data, error = _clean_registration(body)
    if error:
        return error

    try:
        screening = score_screening(body.get("screening"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if screening["tier"] == "red":
        return jsonify({
            "error": "Based on your answers, please speak with a counselor before registering. Nothing was saved.",
            "tier": "red",
        }), 403

    try:
        pre_skill = score_pre_skill(body.get("pre_skill"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        duplicate = _duplicate_error(data["username"], data["email"])
        if duplicate:
            return duplicate
        student_store.create_student(
            data["username"], data["email"], data["password"], data["gender"]
        )
    except DuplicateKeyError:
        return jsonify({"error": "Username or email already exists"}), 409
    except PyMongoError:
        return jsonify({"error": "Could not create the student account."}), 503

    created_id = None
    student_id = None
    stored = None
    try:
        stored = student_store.ensure_student_id(student_store.find_by_username(data["username"]))
        created_id = str(stored["_id"])
        student_id = stored.get("student_id", created_id)

        assessment_store.save_screening(student_id, screening)
        assessment_store.save_pre_skill(student_id, pre_skill)
    except PyMongoError as exc:
        app.logger.error("Registration save failed, rolling back: %s", exc)
        # Run each cleanup on its own so one failure can't skip the other.
        for cleanup in (
            lambda: assessment_store.delete_for_student(student_id) if student_id else None,
            lambda: student_store.delete_user(created_id) if created_id else None,
        ):
            try:
                cleanup()
            except PyMongoError:
                pass
        return jsonify({"error": "Could not complete registration. Please try again."}), 503

    return jsonify({
        "message": "Registration successful.",
        "tier": screening["tier"],   # "green" or "amber" only, no scores
        "access_token": create_access_token(stored),
        "student": {
            "id": created_id,
            "student_id": student_id,
            "username": stored["username"],
            "email": stored["email"],
            "role": stored.get("role", "student"),
            "gender": stored.get("gender"),
        },
    }), 201


@app.post("/api/auth/login")
def login_student():
    body = request.get_json(silent=True) or {}

    username = (body.get("username") or "").strip().lower()
    password = body.get("password") or ""

    if not username or not password:
        return jsonify({
            "error": "Username and password are required."
        }), 400

    try:
        student = student_store.find_by_username(username)

        if not student:
            return jsonify({
                "error": "Invalid username or password."
            }), 401

        student = student_store.ensure_student_id(student)

        valid_password = bcrypt.checkpw(
            password.encode("utf-8"),
            student["password_hash"].encode("utf-8"),
        )

        if not valid_password:
            return jsonify({
                "error": "Invalid username or password."
            }), 401

        if student.get("account_status") == "on_hold":
            return jsonify({
                "error": "Your account is on hold until you have spoken with a counselor."
            }), 403

        token = create_access_token(student)

        return jsonify({
            "message": "Login successful.",
            "access_token": token,
            "student": {
                "id": str(student["_id"]),
                "student_id": student.get("student_id", str(student["_id"])),
                "name": student.get("name"),
                "username": student["username"],
                "email": student["email"],
                "role": student.get("role", "student"),
                **({"gender": student.get("gender")} if student.get("role", "student") == "student" else {}),
                **({"name": student["name"]} if student.get("name") else {}),
                **({"age_group": student["age_group"]} if student.get("age_group") else {}),
            },
        }), 200

    except PyMongoError:
        return jsonify({
            "error": "Could not complete login."
        }), 503


@app.get("/api/auth/me")
@require_auth
def get_current_student():
    try:
        student = g.current_user

        if not student:
            return jsonify({
                "error": "Student account was not found."
            }), 404

        return jsonify({"student": public_user(student)}), 200

    except PyMongoError:
        return jsonify({
            "error": "Could not load student details."
        }), 503


@app.get("/api/users/me")
@require_auth
def get_my_user():
    return get_current_student()


@app.post("/api/auth/logout")
def logout():
    return jsonify({"message": "Logged out successfully."}), 200


def public_user(student):
    user = {
        "id": str(student["_id"]),
        "name": student["name"],
        "username": student["username"],
        "email": student["email"],
        "role": student.get("role", "student"),
    }
    if student.get("name"):
        user["name"] = student["name"]
    if user["role"] == "student":
        user["gender"] = student.get("gender")
        if student.get("age_group"):
            user["age_group"] = student["age_group"]
    return user


@app.get("/api/users")
@require_role("admin")
def list_users():
    try:
        return jsonify({"users": [public_user(user) for user in student_store.list_users()]}), 200
    except PyMongoError:
        return jsonify({"error": "Could not load users."}), 503


@app.get("/api/users/<user_id>")
@require_auth
def get_user(user_id):
    if user_id != g.user_id and g.user_role != "admin":
        return jsonify({"error": "You do not have permission to view this user."}), 403

    try:
        student = student_store.find_by_id(user_id)
        if not student:
            return jsonify({"error": "User was not found."}), 404
        return jsonify({"user": public_user(student)}), 200
    except InvalidId:
        return jsonify({"error": "Invalid user ID."}), 400
    except PyMongoError:
        return jsonify({"error": "Could not load user details."}), 503


def update_user_profile(user_id):
    body = request.get_json(silent=True) or {}
    changes = {}
    try:
        target_user = student_store.find_by_id(user_id)
    except InvalidId:
        return jsonify({"error": "Invalid user ID."}), 400
    if not target_user:
        return jsonify({"error": "User was not found."}), 404

    if "name" in body:
        name = (body.get("name") or "").strip()
        if not name:
            return jsonify({"error": "Name cannot be empty."}), 400
        changes["name"] = name

    if "email" in body:
        email = (body.get("email") or "").strip().lower()
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            return jsonify({"error": "Please provide a valid email address."}), 400
        changes["email"] = email

    if "gender" in body:
        gender = (body.get("gender") or "").strip()
        if target_user.get("role") != "student" or gender not in VALID_GENDERS:
            return jsonify({"error": "Invalid gender"}), 400
        changes["gender"] = gender

    if "age_group" in body:
        age_group = (body.get("age_group") or "").strip()
        if target_user.get("role") != "student" or age_group not in VALID_AGE_GROUPS:
            return jsonify({"error": "Invalid age group"}), 400
        changes["age_group"] = age_group

    forbidden = {"role", "password", "password_hash", "created_at", "_id"}
    if forbidden.intersection(body):
        return jsonify({"error": "Protected user fields cannot be changed."}), 400

    if not changes:
        return jsonify({"error": "At least one profile field is required."}), 400

    try:
        student_store.update_user(user_id, changes)
        return jsonify({"user": public_user(student_store.find_by_id(user_id))}), 200
    except DuplicateKeyError:
        return jsonify({"error": "That email address is already registered."}), 409
    except InvalidId:
        return jsonify({"error": "Invalid user ID."}), 400
    except PyMongoError:
        return jsonify({"error": "Could not update user profile."}), 503


@app.put("/api/users/<user_id>")
@require_auth
def update_user(user_id):
    if user_id != g.user_id and g.user_role != "admin":
        return jsonify({"error": "You do not have permission to update this user."}), 403
    return update_user_profile(user_id)


@app.put("/api/profile")
@require_auth
def update_profile():
    return update_user_profile(g.user_id)


@app.get("/api/profile")
@require_auth
def get_profile():
    return get_current_student()


@app.delete("/api/users/<user_id>")
@require_role("admin")
def delete_user(user_id):
    try:
        if not student_store.delete_user(user_id):
            return jsonify({"error": "User was not found."}), 404
        return jsonify({"message": "User deleted successfully."}), 200
    except InvalidId:
        return jsonify({"error": "Invalid user ID."}), 400
    except PyMongoError:
        return jsonify({"error": "Could not delete user."}), 503


@app.post("/api/users/admin")
@require_role("admin")
def create_admin():
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    username = (body.get("username") or "").strip().lower()
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""

    for field, value in (("Name", name), ("Email", email), ("Username", username), ("Password", password)):
        if not value:
            return jsonify({"error": f"{field} is required"}), 400
    if not re.fullmatch(r"[a-z0-9_]{3,30}", username):
        return jsonify({"error": "Invalid username"}), 400
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        return jsonify({"error": "Please provide a valid email address."}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must contain at least 8 characters."}), 400
    if "gender" in body or "age_group" in body:
        return jsonify({"error": "Admin accounts do not accept student fields."}), 400

    try:
        return jsonify({"message": "Admin created successfully.", "user": student_store.create_admin(name, username, email, password)}), 201
    except DuplicateKeyError:
        duplicate = "Email already exists" if student_store.find_by_email(email) else "Username already exists"
        return jsonify({"error": duplicate}), 409
    except PyMongoError:
        return jsonify({"error": "Could not create the admin account."}), 503


# ---------------------------------------------------
# General support chat
# ---------------------------------------------------
@app.post("/api/general-empathy-chat")
@require_auth
def general_empathy_chat():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "A JSON object is required."}), 400

    message = body.get("message")
    if not isinstance(message, str):
        return jsonify({"error": "Please enter a message."}), 400
    message = message.strip()
    if not message:
        return jsonify({"error": "Please enter a message."}), 400
    if len(message) > 4000:
        return jsonify({"error": "Messages must be 4000 characters or fewer."}), 400

    history = body.get("history", [])
    if not isinstance(history, list) or len(history) > 20:
        return jsonify({"error": "Conversation history must contain at most 20 messages."}), 400
    for entry in history:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("role"), str)
            or entry["role"] not in {"user", "assistant"}
            or not isinstance(entry.get("content"), str)
            or not entry["content"].strip()
            or len(entry["content"]) > 4000
        ):
            return jsonify({
                "error": "Each history message must have a user or assistant role and non-empty content of 4000 characters or fewer."
            }), 400

    try:
        response = generate_empathy_response(message, history)
        return jsonify({"response": response}), 200
    except EmpathyServiceError:
        app.logger.warning("General empathy-support response generation failed.")
        return jsonify({
            "error": "The empathy-support service is currently unavailable."
        }), 503


# ---------------------------------------------------
# Student learning APIs
# ---------------------------------------------------
@app.post("/api/learning-response")
@require_auth
def learning_response():
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    answer_style = body.get("answer_style", "short")
    if not isinstance(answer_style, str) or answer_style not in {"short", "long"}:
        return jsonify({"error": "Invalid answer style."}), 400

    if not question:
        return jsonify({
            "error": "Please enter a question."
        }), 400

    if is_greeting(question):
        return jsonify({
            "educational_response": (
                "Hey! 👋 I'm here to help you explore empathy and build your skills. "
                "What would you like to learn about today?"
            ),
            "learning_context": {
                "message": (
                    "Hey! 👋 I'm here to help you explore empathy and build your skills. "
                    "What would you like to learn about today?"
                ),
                "detected_emotion": "calm",
                "status": "greeting",
                "skill": None,
                "topic": None,
                "answer_style": answer_style,
            },
            "steps": [],
            "suggested_questions": [],
        })

    # Do not accept student_id from frontend.
    # The ID is securely read from the JWT token.
    student_id = g.student_id

    try:
        progress_documents = learning_store.get_all_progress(student_id)
    except PyMongoError:
        app.logger.warning("learning-response progress lookup unavailable.")
        progress_documents = None

    result = answer_student_question(
        question,
        knowledge_base,
        keyword_skill_map,
        answer_style,
        progress_documents=progress_documents,
        progress_structure=progress_service.get_structure(),
    )

    learning_context = result.get("learning_context", result)
    result.setdefault("suggested_questions", [])
    attach_progress_item_id(learning_context)
    app.logger.info(
        "learning-response recommendation skill=%s activity_id=%s",
        (learning_context.get("skill") or {}).get("id"),
        (learning_context.get("recommended_activity") or {}).get("id"),
    )
    steps = result.get("steps", [])
    educational_response = result.get(
        "educational_response",
        result.get("answer", steps if steps else learning_context.get("message", "")),
    )

    try:
        interaction_id = learning_store.save_interaction(
            student_id,
            learning_context,
            educational_response,
            learning_steps=steps,
        )

        result["interaction_id"] = interaction_id
        recommended_activity = learning_context.get("recommended_activity") or {}
        progress_item_id = recommended_activity.get("progress_item_id")
        if progress_item_id:
            learning_store.record_learning_event(
                student_id,
                learning_context["skill"]["id"],
                progress_item_id,
                objective_id=recommended_activity.get("progress_objective_id"),
                source="chatbot",
                status="introduced",
            )
        return jsonify(result), 200

    except PyMongoError:
        return jsonify({
            "error": "Could not save the learning interaction."
        }), 503


@app.get("/api/learning-history")
@require_auth
def learning_history():
    try:
        return jsonify({
            "interactions": learning_store.list_interactions(g.student_id),
        }), 200
    except PyMongoError:
        return jsonify({
            "error": "Could not load learning history."
        }), 503


@app.get("/api/progress/<skill_id>")
@require_auth
def student_progress(skill_id):
    student_id = g.student_id

    try:
        progress = learning_store.get_progress(student_id, skill_id)

        calculated = progress_service.calculate([progress])
        skill = next(
            (item for item in calculated["skills"] if item["skill_id"] == skill_id),
            None,
        )
        verified_completed_ids = {
            objective["objective_id"]
            for objective in (skill or {}).get("objectives", [])
            if objective["completed"]
        }
        progress["next_recommended_learning_objective"] = (
            next_uncompleted_objective(skill_id, verified_completed_ids)
        )
        return jsonify({
            **progress,
            "skill": skill,
            "objectives": skill["objectives"] if skill else [],
        }), 200

    except PyMongoError:
        return jsonify({
            "error": "Could not load student progress."
        }), 503


@app.post("/api/progress/item-complete")
@require_auth
def complete_learning_item():
    body = request.get_json(silent=True) or {}
    skill_value = (
        body.get("skill_id")
        or body.get("skillId")
        or body.get("chapter_id")
        or body.get("chapterId")
    )
    item_value = (
        body.get("item_id")
        or body.get("itemId")
        or body.get("activity_id")
        or body.get("activityId")
        or body.get("video_id")
        or body.get("videoId")
        or body.get("quiz_id")
        or body.get("quizId")
    )
    skill_id = str(skill_value).strip() if skill_value is not None else ""
    item_id = str(item_value).strip() if item_value is not None else ""
    skill_id, item_id = resolve_legacy_progress_ids(skill_id, item_id)
    quiz_score = body.get("quiz_score")
    source = str(body.get("source") or "skills_page").strip().lower()
    requested_item_type = str(body.get("item_type") or "").strip().lower()
    app.logger.info(
        "progress item-complete request student_id=%s body=%s parsed_skill_id=%s parsed_item_id=%s item_type=%s source=%s",
        g.student_id,
        body,
        skill_id,
        item_id,
        body.get("item_type"),
        source,
    )
    if source not in {"skills_page", "chatbot"}:
        return jsonify({"error": "source must be skills_page or chatbot."}), 400

    if not skill_id and item_id:
        matching_chapters = [
            chapter
            for chapter in progress_service.structure.get("chapters", [])
            if any(
                item.get("item_id") == item_id
                for objective in chapter.get("objectives", [])
                for item in objective.get("items", [])
            )
        ]
        if len(matching_chapters) == 1:
            skill_id = matching_chapters[0].get("chapter_id", "")

    chapter = next(
        (
            chapter
            for chapter in progress_service.structure.get("chapters", [])
            if chapter.get("chapter_id") == skill_id
        ),
        None,
    )
    assigned_item = next(
        (
            item
            for objective in (chapter or {}).get("objectives", [])
            for item in objective.get("items", [])
            if item.get("item_id") == item_id
        ),
        None,
    )
    assigned_objective = next(
        (
            objective
            for objective in (chapter or {}).get("objectives", [])
            if any(item.get("item_id") == item_id for item in objective.get("items", []))
        ),
        None,
    )

    if not chapter or not assigned_item:
        app.logger.warning(
            "progress item-complete rejected student_id=%s skill_id=%s item_id=%s",
            g.student_id,
            skill_id,
            item_id,
        )
        return jsonify({
            "error": "A valid assigned skill_id and item_id are required.",
            "received_skill_id": skill_id,
            "received_item_id": item_id,
        }), 400

    requested_objective_id = body.get("objective_id")
    if (
        requested_objective_id is not None
        and requested_objective_id != (assigned_objective or {}).get("objective_id")
    ):
        return jsonify({
            "error": "objective_id does not match the assigned curriculum item.",
        }), 400

    assigned_item_type = str(assigned_item.get("type") or "").lower()
    if assigned_item_type == "practice":
        assigned_item_type = "activity"
    if assigned_item_type == "assessment":
        assigned_item_type = "quiz"
    if requested_item_type and requested_item_type == "practice":
        requested_item_type = "activity"
    if requested_item_type == "assessment":
        requested_item_type = "quiz"
    if requested_item_type and requested_item_type != assigned_item_type:
        return jsonify({
            "error": "item_type does not match the assigned curriculum item.",
            "received_item_type": requested_item_type,
            "assigned_item_type": assigned_item_type,
        }), 400

    if quiz_score is not None:
        if isinstance(quiz_score, bool) or not isinstance(quiz_score, (int, float)):
            return jsonify({"error": "quiz_score must be a number."}), 400
        if quiz_score < 0 or quiz_score > 100:
            return jsonify({"error": "quiz_score must be between 0 and 100."}), 400

    is_quiz = assigned_item_type == "quiz"
    try:
        objective_id = (assigned_objective or {}).get("objective_id")
        requirements = (
            progress_service.completion_requirements(assigned_objective)
            if assigned_objective
            else {"activity_item_ids": [], "quiz_passing_scores": {}}
        )
        passing_score = requirements["quiz_passing_scores"].get(item_id)
        current = learning_store.get_progress(g.student_id, skill_id)

        if is_quiz and quiz_score is None:
            return jsonify({
                "error": "A quiz score is required to record a quiz attempt.",
                "completed": False,
            }), 422

        if is_quiz and quiz_score < passing_score:
            progress = learning_store.record_quiz_attempt(
                g.student_id,
                skill_id,
                item_id,
                quiz_score,
                objective_id=objective_id,
                source=source,
            )
            attempt_passed = False
        else:
            completed_item_ids = set(current.get("completed_item_ids", []))
            completed_item_ids.add(item_id)
            quiz_scores = dict(current.get("quiz_scores", {}))
            if is_quiz:
                previous_score = quiz_scores.get(item_id)
                if isinstance(previous_score, (int, float)) and not isinstance(
                    previous_score, bool
                ):
                    quiz_scores[item_id] = max(previous_score, quiz_score)
                else:
                    quiz_scores[item_id] = quiz_score
            current_state = progress_service.completion_state(
                assigned_objective,
                completed_item_ids,
                quiz_scores,
            )
            completed_objective_ids = set(
                current.get("completed_objective_ids", [])
            )
            if current_state["completed"]:
                completed_objective_ids.add(objective_id)
            next_objective = next_uncompleted_objective(
                skill_id,
                completed_objective_ids,
            )
            progress = learning_store.complete_item(
                g.student_id,
                skill_id,
                item_id,
                objective_id=objective_id,
                next_objective=next_objective,
                objective_completion_requirements=requirements,
                quiz_score=quiz_score,
                source=source,
            )
            attempt_passed = True if is_quiz else None

        calculated = progress_service.calculate([progress])
        skill_summary = next(
            (
                item for item in calculated["skills"]
                if item["skill_id"] == skill_id
            ),
            None,
        )
        objective_summary = next(
            (
                item for item in (skill_summary or {}).get("objectives", [])
                if item["objective_id"] == objective_id
            ),
            None,
        )
        verified_completed_ids = {
            item["objective_id"]
            for item in (skill_summary or {}).get("objectives", [])
            if item["completed"]
        }
        next_objective = next_uncompleted_objective(
            skill_id,
            verified_completed_ids,
        )
        progress["next_recommended_learning_objective"] = next_objective
        app.logger.info(
            "progress item-complete persisted student_id=%s skill_id=%s item_id=%s source=%s completed_item_ids=%s",
            g.student_id,
            skill_id,
            item_id,
            source,
            progress.get("completed_item_ids"),
        )
        missing_conditions = []
        if objective_summary and not objective_summary["activity_completed"]:
            missing_conditions.append("activity")
        if objective_summary and not objective_summary["quiz_passed"]:
            missing_conditions.append("quiz")
        missing_completion_items = []
        if objective_summary and assigned_objective:
            missing_ids = set(
                objective_summary["missing_required_item_ids"]
            )
            missing_completion_items = [
                {
                    "item_id": item.get("item_id"),
                    "type": (
                        "activity"
                        if str(item.get("type") or "").lower() == "practice"
                        else str(item.get("type") or "").lower()
                    ),
                    "title": item.get("title") or item.get("item_id"),
                }
                for item in assigned_objective.get("items", [])
                if item.get("item_id") in missing_ids
            ]
        return jsonify({
            **progress,
            "quiz_passed": bool(
                objective_summary and objective_summary["quiz_passed"]
            ),
            "quiz_attempt_passed": attempt_passed,
            "passing_score": passing_score,
            "objective_completed": bool(
                objective_summary and objective_summary["completed"]
            ),
            "completed_objective": (
                {
                    "id": objective_id,
                    "title": assigned_objective.get("title", ""),
                }
                if objective_summary and objective_summary["completed"]
                else None
            ),
            "next_incomplete_objective": next_objective,
            "missing_completion_conditions": missing_conditions,
            "missing_completion_items": missing_completion_items,
        }), 200
    except PyMongoError:
        return jsonify({
            "error": "Could not update student progress."
        }), 503


@app.get("/api/progress")
@require_auth
def all_student_progress():
    try:
        progress = learning_store.get_all_progress(g.student_id)
        return jsonify(progress_service.calculate(progress)), 200
    except PyMongoError:
        return jsonify({
            "error": "Could not load student progress."
        }), 503


@app.get("/api/progress/structure")
@require_auth
def progress_structure():
    try:
        return jsonify(progress_service.get_structure()), 200
    except Exception:
        return jsonify({
            "error": "Could not load progress structure."
        }), 500


@app.post("/api/objectives/<objective_id>/complete")
@require_auth
def complete_objective(objective_id):
    student_id = g.student_id

    body = request.get_json(silent=True) or {}
    skill_id = (body.get("skill_id") or "").strip()
    chapter = next(
        (
            chapter
            for chapter in progress_service.structure.get("chapters", [])
            if chapter.get("chapter_id") == skill_id
        ),
        None,
    )
    objective = next(
        (
            objective
            for objective in (chapter or {}).get("objectives", [])
            if objective.get("objective_id") == objective_id
        ),
        None,
    )

    if not skill_id or not objective:
        return jsonify({
            "error": "A valid skill_id and learning objective are required."
        }), 400

    try:
        current = learning_store.get_progress(student_id, skill_id)

        state = progress_service.completion_state(
            objective,
            current.get("completed_item_ids", []),
            current.get("quiz_scores", {}),
        )
        chatbot_quick_check_passed = False
        if (
            body.get("source") == "chatbot"
            and isinstance(body.get("quick_check_answer"), bool)
        ):
            interaction = learning_store.get_interaction(
                student_id,
                body.get("interaction_id"),
            )
            selected_skill = (interaction or {}).get("selected_skill") or {}
            interaction_objective = (
                (interaction or {}).get("learning_objective") or {}
            )
            saved_steps = (interaction or {}).get("learning_steps")
            practice_check = next(
                (
                    step
                    for step in (
                        saved_steps if isinstance(saved_steps, list) else []
                    )
                    if isinstance(step, dict)
                    and step.get("type") == "practice_check"
                ),
                {},
            )
            interaction_skill_id = (
                selected_skill.get("id")
                or selected_skill.get("skill_id")
                or selected_skill.get("chapter_id")
            )
            interaction_objective_id = (
                interaction_objective.get("id")
                or interaction_objective.get("objective_id")
            )
            correct_answer = practice_check.get("correct_answer")
            chatbot_quick_check_passed = (
                interaction_skill_id == skill_id
                and interaction_objective_id == objective_id
                and isinstance(correct_answer, bool)
                and body["quick_check_answer"] is correct_answer
            )
        can_complete = state["completed"] or (
            chatbot_quick_check_passed and state["activity_completed"]
        )
        if not can_complete:
            missing_conditions = []
            if not state["activity_completed"]:
                missing_conditions.append("activity")
            if not state["quiz_passed"] and not chatbot_quick_check_passed:
                missing_conditions.append("quiz")
            if (
                body.get("source") == "chatbot"
                and not chatbot_quick_check_passed
            ):
                completion_error = (
                    "A correct, verified understanding check is required "
                    "to complete this objective."
                )
            elif chatbot_quick_check_passed:
                completion_error = (
                    "Complete the objective activity before completing this objective."
                )
            else:
                completion_error = (
                    "Complete the objective activity and pass its quiz before "
                    "completing this objective."
                )
            return jsonify({
                "error": completion_error,
                "completed": False,
                "objective_id": objective_id,
                "required_item_ids": state["required_item_ids"],
                "completed_item_ids": sorted(
                    set(current.get("completed_item_ids", []))
                    & set(state["required_item_ids"])
                ),
                "missing_item_ids": state["missing_required_item_ids"],
                "missing_completion_conditions": missing_conditions,
            }), 422

        calculated = progress_service.calculate([current])
        skill_summary = next(
            (
                item for item in calculated["skills"]
                if item["skill_id"] == skill_id
            ),
            None,
        )
        completed = {
            item["objective_id"]
            for item in (skill_summary or {}).get("objectives", [])
            if item["completed"]
        }
        completed.add(objective_id)
        objective_record = {
            "id": objective["objective_id"],
            "title": objective.get("title", ""),
            "content": objective.get("content", ""),
        }

        progress = learning_store.complete_objective(
            student_id,
            skill_id,
            objective_record,
            next_uncompleted_objective(skill_id, completed),
        )

        return jsonify(progress), 200

    except PyMongoError:
        return jsonify({
            "error": "Could not update student progress."
        }), 503


# ---------------------------------------------------
# Existing reviewer evaluation APIs
# ---------------------------------------------------

@app.post("/api/interactions/<interaction_id>/evaluation")
def evaluate_interaction(interaction_id):
    body = request.get_json(silent=True) or {}

    required = [
        "correct_skill_selected",
        "activity_matches_emotion",
        "response_educationally_aligned",
    ]

    if any(not isinstance(body.get(field), bool) for field in required):
        return jsonify({
            "error": "Each evaluation criterion must be true or false."
        }), 400

    try:
        evaluation_id = learning_store.save_evaluation(
            interaction_id,
            (body.get("reviewer_id") or "unassigned").strip(),
            body,
        )

        return jsonify({
            "evaluation_id": evaluation_id
        }), 201

    except InvalidId:
        return jsonify({
            "error": "Invalid interaction ID."
        }), 400

    except PyMongoError:
        return jsonify({
            "error": "Could not save the evaluation."
        }), 503


@app.get("/api/evaluations/summary")
def evaluation_summary():
    try:
        return jsonify(learning_store.evaluation_summary()), 200

    except PyMongoError:
        return jsonify({
            "error": "Could not calculate the evaluation summary."
        }), 503


if __name__ == "__main__":
    # The Werkzeug reloader can close the listening socket while its serving
    # thread is still polling it on Windows, producing WinError 10038.
    app.run(debug=True, port=5000, use_reloader=False)
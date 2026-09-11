import json
import re
from pathlib import Path

import bcrypt
from flask import Flask, g, jsonify, request
from bson.errors import InvalidId
from pymongo.errors import DuplicateKeyError, PyMongoError
from dotenv import load_dotenv

from auth import VALID_AGE_GROUPS, VALID_GENDERS, create_access_token, get_user_store, require_auth, require_role
from learning.pedagogical_controller import answer_student_question, belongs_to_skill
from learning.interaction_store import LearningStore

from flask_cors import CORS


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

with open(BASE_DIR / "output" / "keyword_skill_map.json", "r", encoding="utf-8") as file:
    keyword_skill_map = json.load(file)

nodes_by_id = {node["id"]: node for node in knowledge_base}

learning_store = LearningStore()
student_store = get_user_store()


def get_skill_objectives(skill_id):
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

    name = (body.get("name") or "").strip()
    username = (body.get("username") or "").strip().lower()
    email = (body.get("email") or "").strip().lower()
    gender = (body.get("gender") or "").strip()
    age_group = (body.get("age_group") or "").strip()
    password = body.get("password") or ""

    for field, value in (("Name", name), ("Email", email), ("Gender", gender), ("Age group", age_group), ("Username", username), ("Password", password)):
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
    if age_group not in VALID_AGE_GROUPS:
        return jsonify({"error": "Invalid age group"}), 400

    try:
        student = student_store.create_student(name, username, email, password, gender, age_group)

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

        valid_password = bcrypt.checkpw(
            password.encode("utf-8"),
            student["password_hash"].encode("utf-8"),
        )

        if not valid_password:
            return jsonify({
                "error": "Invalid username or password."
            }), 401

        token = create_access_token(student)

        return jsonify({
            "message": "Login successful.",
            "access_token": token,
            "student": {
                "id": str(student["_id"]),
                "name": student["name"],
                "username": student["username"],
                "email": student["email"],
                "role": student.get("role", "student"),
                **({"gender": student.get("gender"), "age_group": student.get("age_group")} if student.get("role", "student") == "student" else {}),
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
        student = student_store.find_by_id(g.student_id)

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
    if user["role"] == "student":
        user.update({"gender": student.get("gender"), "age_group": student.get("age_group")})
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
    if user_id != g.student_id and g.user_role != "admin":
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
    if user_id != g.student_id and g.user_role != "admin":
        return jsonify({"error": "You do not have permission to update this user."}), 403
    return update_user_profile(user_id)


@app.put("/api/profile")
@require_auth
def update_profile():
    return update_user_profile(g.student_id)


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
# Student learning APIs
# ---------------------------------------------------

@app.post("/api/learning-response")
@require_auth
def learning_response():
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()

    if not question:
        return jsonify({
            "error": "Please enter a question."
        }), 400

    # Do not accept student_id from frontend.
    # The ID is securely read from the JWT token.
    student_id = g.student_id

    result = answer_student_question(
        question,
        knowledge_base,
        keyword_skill_map,
    )

    learning_context = result.get("learning_context", result)
    steps = result.get("steps", [])
    educational_response = result.get(
        "educational_response",
        steps if steps else learning_context.get("message", ""),
    )

    try:
        interaction_id = learning_store.save_interaction(
            student_id,
            learning_context,
            educational_response,
        )

        result["interaction_id"] = interaction_id
        return jsonify(result), 200

    except PyMongoError:
        return jsonify({
            "error": "Could not save the learning interaction."
        }), 503


@app.get("/api/progress/<skill_id>")
@require_auth
def student_progress(skill_id):
    student_id = g.student_id

    try:
        progress = learning_store.get_progress(student_id, skill_id)

        if progress["next_recommended_learning_objective"] is None:
            progress["next_recommended_learning_objective"] = (
                next_uncompleted_objective(
                    skill_id,
                    progress["completed_objective_ids"],
                )
            )

        return jsonify(progress), 200

    except PyMongoError:
        return jsonify({
            "error": "Could not load student progress."
        }), 503


@app.post("/api/objectives/<objective_id>/complete")
@require_auth
def complete_objective(objective_id):
    student_id = g.student_id

    body = request.get_json(silent=True) or {}
    skill_id = (body.get("skill_id") or "").strip()
    objective = nodes_by_id.get(objective_id)

    if not skill_id or not objective:
        return jsonify({
            "error": "A valid skill_id and learning objective are required."
        }), 400

    if objective.get("type") != "learning_objective":
        return jsonify({
            "error": "This item is not a learning objective."
        }), 400

    if not belongs_to_skill(objective, skill_id, nodes_by_id):
        return jsonify({
            "error": "This objective does not belong to the selected skill."
        }), 400

    try:
        current = learning_store.get_progress(student_id, skill_id)

        completed = set(current["completed_objective_ids"])
        completed.add(objective_id)

        progress = learning_store.complete_objective(
            student_id,
            skill_id,
            objective,
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
    app.run(debug=True, port=5000)
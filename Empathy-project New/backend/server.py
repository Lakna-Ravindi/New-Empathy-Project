import json
from pathlib import Path

import bcrypt
from flask import Flask, g, jsonify, request
from bson.errors import InvalidId
from pymongo.errors import DuplicateKeyError, PyMongoError
from dotenv import load_dotenv

from auth import StudentStore, create_access_token, require_auth
from learning.pedagogical_controller import answer_student_question, belongs_to_skill
from learning.interaction_store import LearningStore


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / "backend" / ".env")

with open(BASE_DIR / "output" / "knowledge_base.json", "r", encoding="utf-8") as file:
    knowledge_base = json.load(file)

with open(BASE_DIR / "output" / "keyword_skill_map.json", "r", encoding="utf-8") as file:
    keyword_skill_map = json.load(file)

nodes_by_id = {node["id"]: node for node in knowledge_base}

learning_store = LearningStore()
student_store = StudentStore()


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
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""

    if not name or not email or not password:
        return jsonify({
            "error": "Name, email, and password are required."
        }), 400

    if "@" not in email:
        return jsonify({
            "error": "Please provide a valid email address."
        }), 400

    if len(password) < 8:
        return jsonify({
            "error": "Password must contain at least 8 characters."
        }), 400

    try:
        student = student_store.create_student(name, email, password)

        # Login token is returned immediately after registration.
        stored_student = student_store.find_by_email(email)
        token = create_access_token(stored_student)

        return jsonify({
            "message": "Registration successful.",
            "student": student,
            "access_token": token,
        }), 201

    except DuplicateKeyError:
        return jsonify({
            "error": "An account already exists with this email."
        }), 409

    except PyMongoError:
        return jsonify({
            "error": "Could not create the student account."
        }), 503


@app.post("/api/auth/login")
def login_student():
    body = request.get_json(silent=True) or {}

    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""

    if not email or not password:
        return jsonify({
            "error": "Email and password are required."
        }), 400

    try:
        student = student_store.find_by_email(email)

        if not student:
            return jsonify({
                "error": "Invalid email or password."
            }), 401

        valid_password = bcrypt.checkpw(
            password.encode("utf-8"),
            student["password_hash"].encode("utf-8"),
        )

        if not valid_password:
            return jsonify({
                "error": "Invalid email or password."
            }), 401

        token = create_access_token(student)

        return jsonify({
            "message": "Login successful.",
            "access_token": token,
            "student": {
                "id": str(student["_id"]),
                "name": student["name"],
                "email": student["email"],
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

        return jsonify({
            "student": {
                "id": str(student["_id"]),
                "name": student["name"],
                "email": student["email"],
            }
        }), 200

    except PyMongoError:
        return jsonify({
            "error": "Could not load student details."
        }), 503


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
    educational_response = result.get(
        "educational_response",
        learning_context.get("message", ""),
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
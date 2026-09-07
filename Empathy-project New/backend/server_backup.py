import json
from pathlib import Path

from bson.errors import InvalidId
from pymongo.errors import PyMongoError
from dotenv import load_dotenv
import os
import re
from datetime import datetime, timezone
from functools import wraps

from flask import Flask, request, jsonify, session
from pymongo.errors import DuplicateKeyError
from werkzeug.security import generate_password_hash, check_password_hash

from learning.pedagogical_controller import answer_student_question, belongs_to_skill
from learning.interaction_store import LearningStore

from app.core.security import verify_password, create_access_token, hash_password
from datetime import timedelta



app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY")

# In production, use HTTPS and turn this on.
app.config["SESSION_COOKIE_SECURE"] = False
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# Find the project folder.
BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / "backend" / ".env")

# Load your approved learning content once when the server starts.
with open(BASE_DIR / "output" / "knowledge_base.json", "r", encoding="utf-8") as file:
    knowledge_base = json.load(file)

with open(BASE_DIR / "output" / "keyword_skill_map.json", "r", encoding="utf-8") as file:
    keyword_skill_map = json.load(file)

nodes_by_id = {node["id"]: node for node in knowledge_base}
learning_store = LearningStore()
users = None
if learning_store.db is not None:
    users = learning_store.db["users"]
    users.create_index("email", unique=True)
    users.create_index("username", unique=True)


def get_skill_objectives(skill_id):
    """Return objectives in the approved source order for one skill."""
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
        (objective for objective in get_skill_objectives(skill_id)
         if objective["id"] not in completed_ids),
        None,
    )


def current_time():
    return datetime.now(timezone.utc)


def require_login(view_function):
    @wraps(view_function)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Please log in first."}), 401
        return view_function(*args, **kwargs)
    return wrapped


def public_user(user):
    return {
        "id": str(user["_id"]),
        "name": user["name"],
        "email": user["email"],
        "username": user["username"],
        "age_group": user["age_group"],
        "gender": user["gender"],
        "role": user["role"],
    }
# Since you're using Flask, use this decorator instead of FastAPI's Depends:
def require_jwt(view_function):
    """Decorator to require valid JWT token."""
    @wraps(view_function)
    def wrapped(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401
        
        token = auth_header[7:]  # Remove "Bearer " prefix
        
        from app.core.security import get_user_id_from_token
        from bson import ObjectId
        
        user_id = get_user_id_from_token(token)
        if not user_id:
            return jsonify({"error": "Invalid or expired token"}), 401
        
        try:
            user = users.find_one({"_id": ObjectId(user_id)})
        except:
            user = None
        
        if not user:
            return jsonify({"error": "User not found"}), 401
        
        return view_function(user, *args, **kwargs)
    
    return wrapped


@app.post("/api/auth/register")
def register():
    if users is None:
        return jsonify({"error": "User database is unavailable."}), 503

    body = request.get_json(silent=True) or {}

    name = (body.get("name") or "").strip()
    email = (body.get("email") or "").strip().lower()
    username = (body.get("username") or "").strip().lower()
    password = body.get("password") or ""
    age_group = (body.get("age_group") or "").strip()
    gender = (body.get("gender") or "").strip()

    if not all([name, email, username, password, age_group, gender]):
        return jsonify({"error": "All registration fields are required."}), 400

    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        return jsonify({"error": "Please enter a valid email address."}), 400

    if not re.fullmatch(r"[a-z0-9_]{3,30}", username):
        return jsonify({
            "error": "Username must be 3-30 characters and use only letters, numbers, or underscores."
        }), 400

    if len(password) < 8:
        return jsonify({
            "error": "Password must contain at least 8 characters."
        }), 400

    allowed_age_groups = {
        "under_16",
        "16_18",
        "19_21",
        "22_25",
        "26_plus"
    }

    if age_group not in allowed_age_groups:
        return jsonify({"error": "Please select a valid age group."}), 400

    user = {
        "name": name,
        "email": email,
        "username": username,
        "password_hash": hash_password(password),
        "age_group": age_group,
        "gender": gender,
        "role": "student",
        "created_at": current_time(),
    }

    try:
        result = users.insert_one(user)
    except DuplicateKeyError:
        return jsonify({
            "error": "That email address or username is already registered."
        }), 409

    session["user_id"] = str(result.inserted_id)
    session["username"] = username

    user["_id"] = result.inserted_id

    return jsonify({
        "message": "Registration successful.",
        "user": public_user(user),
    }), 201

@app.post("/api/auth/login")
def login():
    """Login with username/password, return JWT token."""
    if users is None:
        return jsonify({"error": "User database is unavailable."}), 503

    body = request.get_json(silent=True) or {}
    username = (body.get("username") or "").strip().lower()
    password = body.get("password") or ""

    if not username or not password:
        return jsonify({"error": "Username and password are required."}), 400

    user = users.find_one({"username": username})

    # Use JWT-compatible verification
    if user is None or not verify_password(password, user["password_hash"]):
        return jsonify({"error": "Invalid username or password."}), 401

    # Create JWT token with user ID in 'sub'
    access_token = create_access_token(
        data={"sub": str(user["_id"])},
        expires_delta=timedelta(minutes=30)
    )

    return jsonify({
        "message": "Login successful.",
        "access_token": access_token,
        "token_type": "Bearer",
        "user": public_user(user),
    }), 200

# @app.post("/api/auth/login")
# def login():
#     if users is None:
#         return jsonify({"error": "User database is unavailable."}), 503

#     body = request.get_json(silent=True) or {}

#     username = (body.get("username") or "").strip().lower()
#     password = body.get("password") or ""

#     if not username or not password:
#         return jsonify({"error": "Username and password are required."}), 400

#     user = users.find_one({"username": username})

#     if user is None or not check_password_hash(
#         user["password_hash"],
#         password
#     ):
#         return jsonify({"error": "Invalid username or password."}), 401

#     session["user_id"] = str(user["_id"])
#     session["username"] = user["username"]

#     return jsonify({
#         "message": "Login successful.",
#         "user": public_user(user),
#     }), 200

@app.post("/api/auth/logout")
def logout():
    """
    Client-side logout (optional).
    In JWT, logout is typically handled by client deleting the token.
    This endpoint can be used to blacklist tokens if needed.
    """
    return jsonify({"message": "Logged out successfully."}), 200

# @app.post("/api/auth/logout")
# def logout():
#     session.clear()
#     return jsonify({"message": "Logged out successfully."}), 200

# Protected endpoint example
@app.get("/api/me/profile")
@require_jwt
def get_me_profile(current_user):
    """Get current user's profile."""
    return jsonify(public_user(current_user)), 200


@app.get("/api/me/progress/<skill_id>")
@require_jwt
def get_user_progress(current_user, skill_id):
    """Get current user's progress in a skill."""
    user_id = str(current_user["_id"])
    
    # Fetch progress from your learning store
    progress = learning_store.get_user_progress(user_id, skill_id)
    
    return jsonify(progress), 200

@app.post("/api/learning-response")
def learning_response():
    # Receive the student's question from the website.
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()

    # Do not continue if no question was sent.
    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    student_id = (body.get("student_id") or "").strip()
    if not student_id:
        return jsonify({"error": "A student_id is required."}), 400

    # Controller selects the learning path; Gemini writes the wording.
    result = answer_student_question(
        question,
        knowledge_base,
        keyword_skill_map
    )

    learning_context = result.get("learning_context", result)
    educational_response = result.get(
        "educational_response", learning_context.get("message", "")
    )

    try:
        interaction_id = learning_store.save_interaction(
            student_id, learning_context, educational_response
        )
    except PyMongoError as e:
        print("MONGODB SAVE ERROR:", repr(e))
        return jsonify({
            "error": "Could not save the learning interaction.",
            "details": str(e)
        }), 503

    # Send the decision, Gemini's wording, and the audit-record ID to the website.
    result["interaction_id"] = interaction_id
    return jsonify(result), 200


@app.get("/api/students/<student_id>/progress/<skill_id>")
def student_progress(student_id, skill_id):
    try:
        progress = learning_store.get_progress(student_id, skill_id)
        if progress["next_recommended_learning_objective"] is None:
            progress["next_recommended_learning_objective"] = next_uncompleted_objective(
                skill_id, progress["completed_objective_ids"]
            )
        return jsonify(progress), 200
    except PyMongoError:
        return jsonify({"error": "Could not load student progress."}), 503


@app.post("/api/students/<student_id>/objectives/<objective_id>/complete")
def complete_objective(student_id, objective_id):
    body = request.get_json(silent=True) or {}
    skill_id = (body.get("skill_id") or "").strip()
    objective = nodes_by_id.get(objective_id)

    if not skill_id or not objective or objective.get("type") != "learning_objective":
        return jsonify({"error": "A valid skill_id and learning objective are required."}), 400
    if not belongs_to_skill(objective, skill_id, nodes_by_id):
        return jsonify({"error": "This objective does not belong to the selected skill."}), 400

    try:
        current = learning_store.get_progress(student_id, skill_id)
        completed = set(current["completed_objective_ids"]) | {objective_id}
        progress = learning_store.complete_objective(
            student_id,
            skill_id,
            objective,
            next_uncompleted_objective(skill_id, completed),
        )
        return jsonify(progress), 200
    except PyMongoError:
        return jsonify({"error": "Could not update student progress."}), 503


@app.post("/api/interactions/<interaction_id>/evaluation")
def evaluate_interaction(interaction_id):
    body = request.get_json(silent=True) or {}
    required = [
        "correct_skill_selected",
        "activity_matches_emotion",
        "response_educationally_aligned",
    ]

    if any(not isinstance(body.get(field), bool) for field in required):
        return jsonify({"error": "Each evaluation criterion must be true or false."}), 400

    try:
        evaluation_id = learning_store.save_evaluation(
            interaction_id,
            (body.get("reviewer_id") or "unassigned").strip(),
            body,
        )
        return jsonify({"evaluation_id": evaluation_id}), 201
    except InvalidId:
        return jsonify({"error": "Invalid interaction ID."}), 400
    except PyMongoError:
        return jsonify({"error": "Could not save the evaluation."}), 503


@app.get("/api/evaluations/summary")
def evaluation_summary():
    try:
        return jsonify(learning_store.evaluation_summary()), 200
    except PyMongoError:
        return jsonify({"error": "Could not calculate the evaluation summary."}), 503


if __name__ == "__main__":
    app.run(debug=True, port=5000)

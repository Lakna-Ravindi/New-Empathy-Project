import json
from pathlib import Path

from flask import Flask, request, jsonify
from bson.errors import InvalidId
from pymongo.errors import PyMongoError
from dotenv import load_dotenv

from learning.pedagogical_controller import answer_student_question, belongs_to_skill
from learning.interaction_store import LearningStore


app = Flask(__name__)

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

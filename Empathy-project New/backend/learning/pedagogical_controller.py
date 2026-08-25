import json
import re
import sys
from pathlib import Path

from learning.gemini_responder import generate_educational_response


# Simple concept expansion for the first prototype.
# You can add more words after testing.
CONCEPT_ALIASES = {
    "overwhelmed": [
        "stress",
        "anxiety",
        "grounding",
        "resilient zone",
        "dysregulation",
        "calm"
    ],
    "anxious": [
        "anxiety",
        "grounding",
        "resilient zone",
        "breathing",
        "calm"
    ],
    "angry": [
        "anger",
        "emotional regulation",
        "resilient zone",
        "self-control"
    ]
}

# Rule-based metadata used for progress records. It does not make or override
# the controller's learning decision.
EMOTION_KEYWORDS = {
    "anxious": {"anxious", "anxiety", "overwhelmed", "stress", "stressed", "nervous", "worried", "worry"},
    "angry": {"angry", "anger", "mad", "frustrated", "frustration"},
    "sad": {"sad", "upset", "lonely", "down", "grief"},
    "calm": {"calm", "relaxed", "peaceful"},
}

STOP_WORDS = {
    "i", "am", "is", "are", "the", "a", "an", "and", "or",
    "to", "of", "in", "on", "for", "with", "before", "after",
    "what", "how", "can", "do", "feel", "my", "me", "it"
}


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_text(text):
    text = (text or "").lower()
    return re.sub(r"\s+", " ", text).strip()


def get_terms(text):
    words = re.findall(r"[a-zA-Z]+", normalize_text(text))
    return {word for word in words if word not in STOP_WORDS}


def expand_terms(question):
    terms = get_terms(question)

    for word in list(terms):
        for related_term in CONCEPT_ALIASES.get(word, []):
            terms.update(get_terms(related_term))

    return terms


def detect_emotion(question):
    """Return the strongest emotion signal found in a student question."""
    terms = get_terms(question)
    scores = {
        emotion: len(terms.intersection(keywords))
        for emotion, keywords in EMOTION_KEYWORDS.items()
    }
    emotion, score = max(scores.items(), key=lambda item: item[1])
    return emotion if score else "unknown"


def score_text(text, query_terms):
    """Calculate a simple keyword-overlap score."""
    text_terms = get_terms(text)

    if not text_terms:
        return 0

    return len(text_terms.intersection(query_terms))


def find_skill_mapping(question, keyword_skill_map):
    """
    Find the best Skill-Keyword record for the student's question.
    """
    query_terms = expand_terms(question)
    best_mapping = None
    best_score = 0

    for item in keyword_skill_map:
        if item.get("mapping_status") != "mapped":
            continue

        searchable_text = " ".join([
            item.get("keyword") or "",
            item.get("source_node_title") or "",
            item.get("topic") or "",
            item.get("skill") or ""
        ])

        score = score_text(searchable_text, query_terms)

        if score > best_score:
            best_score = score
            best_mapping = item

    return best_mapping, best_score, query_terms


def belongs_to_skill(node, skill_id, nodes_by_id):
    """Check whether a node belongs to a specific chapter / skill."""
    current = node

    while current:
        if current.get("id") == skill_id:
            return True

        parent_id = current.get("parent_id")
        current = nodes_by_id.get(parent_id)

    return False


def select_learning_objective(skill_id, nodes, nodes_by_id, query_terms):
    """
    Select the most relevant learning-objective node inside the skill.
    """
    objectives = [
        node for node in nodes
        if node.get("type") == "learning_objective"
        and belongs_to_skill(node, skill_id, nodes_by_id)
    ]

    if not objectives:
        return None

    return max(
        objectives,
        key=lambda node: score_text(
            f"{node.get('title', '')} {node.get('content', '')}",
            query_terms
        )
    )


def select_learning_activity(skill_id, nodes, nodes_by_id, query_terms):
    """
    Select a relevant practice, activity, or reflection from the same skill.
    """
    allowed_types = {"practice", "activity", "reflection"}

    activities = [
        node for node in nodes
        if node.get("type") in allowed_types
        and belongs_to_skill(node, skill_id, nodes_by_id)
    ]

    if not activities:
        return None

    # Prefer the activity with the strongest connection to the question.
    return max(
        activities,
        key=lambda node: score_text(
            f"{node.get('title', '')} {node.get('content', '')}",
            query_terms
        )
    )


def process_student_question(question, knowledge_base, keyword_skill_map):
    """
    Main Pedagogical Controller function.

    Student question
      → identify keywords / skill
      → select learning objective
      → select activity
      → return a structured learning decision
    """
    nodes_by_id = {
        node["id"]: node
        for node in knowledge_base
    }

    mapping, score, query_terms = find_skill_mapping(
        question,
        keyword_skill_map
    )

    if mapping is None or score == 0:
        return {
            "status": "no_match",
            "student_question": question,
            "message": (
                "I could not find a suitable topic in the approved "
                "SEEK learning content. Please choose an empathy skill "
                "or ask about a related learning topic."
            )
        }

    skill_id = mapping["skill_id"]

    objective = select_learning_objective(
        skill_id,
        knowledge_base,
        nodes_by_id,
        query_terms
    )

    activity = select_learning_activity(
        skill_id,
        knowledge_base,
        nodes_by_id,
        query_terms
    )

    return {
        "status": "learning_path_selected",
        "student_question": question,
        "detected_emotion": detect_emotion(question),

        "detected_terms": sorted(query_terms),

        "matched_keyword": mapping["keyword"],

        "skill": {
            "id": mapping["skill_id"],
            "title": mapping["skill"]
        },

        "topic": {
            "id": mapping.get("topic_id"),
            "title": mapping.get("topic")
        },

        "learning_objective": {
            "id": objective["id"] if objective else None,
            "title": objective["title"] if objective else None,
            "content": objective["content"] if objective else None
        },

        "recommended_activity": {
            "id": activity["id"] if activity else None,
            "type": activity["type"] if activity else None,
            "title": activity["title"] if activity else None,
            "content": activity["content"] if activity else None,
            "page": activity["page"] if activity else None
        },

        "source_page": mapping["page"]
    }

def answer_student_question(question, knowledge_base, keyword_skill_map):
    # 1. Your Pedagogical Controller makes all learning decisions.
    learning_context = process_student_question(
        question,
        knowledge_base,
        keyword_skill_map
    )

    # 2. If the controller could not find an approved lesson,
    #    return its own message. Gemini is not called.
    if learning_context["status"] != "learning_path_selected":
        return learning_context

    # 3. Gemini receives only the controller's selected lesson.
    gemini_response = generate_educational_response(learning_context)

    # 4. Return both the decision and the friendly response.
    return {
        "learning_context": learning_context,
        "educational_response": gemini_response
    }


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]

    knowledge_base = load_json(
        BASE_DIR / "output" / "knowledge_base.json"
    )

    keyword_skill_map = load_json(
        BASE_DIR / "output" / "keyword_skill_map.json"
    )

    question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "I feel overwhelmed before exams. How can I calm down?"
    )

    result = answer_student_question(
        question,
        knowledge_base,
        keyword_skill_map
    )

    print(json.dumps(result, indent=4, ensure_ascii=False))

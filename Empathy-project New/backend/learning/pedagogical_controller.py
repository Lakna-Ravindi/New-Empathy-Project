import json
import re
import sys
from pathlib import Path

from learning.gemini_responder import generate_educational_response

VALID_ANSWER_STYLES = {"short", "long"}
QUESTION_RELEVANT_CONTENT_LIMIT = 5


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
    "what", "how", "can", "do", "feel", "my", "me", "it",
    "about", "learn", "want", "need", "down"
}

# Deliberately limited to common forms needed by the controller. This avoids
# broad stemming that could merge unrelated concepts or skill mappings.
WORD_FORM_NORMALIZATIONS = {
    "exams": "exam",
    "calming": "calm",
    "staying": "stay",
    "stressed": "stress",
}

MEANINGFUL_CONCEPTS = {
    "anxiety", "anxious", "anger", "angry", "calm", "compassion",
    "dysregulation", "emotional", "empathy", "forgiveness",
    "grounding", "mindfulness", "regulation", "resilient", "self",
    "stress", "tracking",
}

EXPLICIT_CONCEPT_PHRASES = {
    "emotional regulation",
    "resilient zone",
    "self control",
}


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_text(text):
    text = (text or "").lower()
    return re.sub(r"\s+", " ", text).strip()


def _normalize_word(word):
    return WORD_FORM_NORMALIZATIONS.get(word, word)


def normalize_keyword(text):
    """Return a stable, punctuation-insensitive mapping label."""
    return " ".join(
        normalized_word
        for word in re.findall(r"[a-z]+", normalize_text(text))
        if word not in STOP_WORDS and len(word) > 1
        for normalized_word in [_normalize_word(word)]
    )


def get_terms(text):
    words = re.findall(r"[a-zA-Z]+", normalize_text(text))
    return {
        _normalize_word(word) for word in words
        if word not in STOP_WORDS and len(word) > 1
    }


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


def requests_actionable_help(question):
    """Identify questions where a second permission prompt would be repetitive."""
    normalized = normalize_text(question)
    return bool(
        re.search(r"\b(how can|what can|what should|help me|how do i)\b", normalized)
        or re.search(r"\b(need help|want help)\b", normalized)
    )


def score_text(text, query_terms):
    """Calculate a simple keyword-overlap score."""
    text_terms = get_terms(text)

    if not text_terms:
        return 0

    exact_matches = text_terms.intersection(query_terms)
    related_matches = {
        query_term
        for query_term in query_terms.intersection(MEANINGFUL_CONCEPTS)
        if any(
            text_term.startswith(query_term) or query_term.startswith(text_term)
            for text_term in text_terms
        )
    }
    return len(exact_matches.union(related_matches))


def keyword_match_score(keyword, question, detected_terms, query_terms):
    """Score a mapping keyword without making every alias an exact match."""
    keyword_text = normalize_keyword(keyword)
    keyword_terms = get_terms(keyword_text)
    question_terms = get_terms(question)

    if not keyword_terms:
        return 0

    direct_match = keyword_terms.issubset(question_terms)
    direct_concepts = keyword_terms.intersection(question_terms).intersection(
        MEANINGFUL_CONCEPTS
    )
    partial_concept_match = bool(direct_concepts)

    overlap = keyword_terms.intersection(query_terms)
    expanded_match = bool(overlap) and (
        len(keyword_terms) == 1
        or len(overlap) == len(keyword_terms)
    )

    if not direct_match and not expanded_match and not partial_concept_match:
        return 0

    score = len(keyword_terms.intersection(detected_terms)) * 100
    if partial_concept_match and not direct_match:
        score = len(direct_concepts) * 80
    if expanded_match and not direct_match:
        score += len(keyword_terms.intersection(query_terms)) * 10
    return score


def is_relevant_mapping_keyword(keyword, source_node_type=None):
    """Reject extracted content sentences masquerading as mapping keywords."""
    normalized_keyword = normalize_keyword(keyword)
    terms = get_terms(keyword)
    meaningful_terms = terms.intersection(MEANINGFUL_CONCEPTS)
    if not meaningful_terms or len(terms) > 4:
        return False
    if len(terms) == 1:
        return True
    return (
        normalized_keyword in EXPLICIT_CONCEPT_PHRASES
        or source_node_type != "content"
    )


def find_skill_mapping(question, keyword_skill_map):
    """
    Find every mapped Skill-Keyword record relevant to the student's question.

    The return shape keeps the former tuple contract for callers while the
    first item is now a de-duplicated list ordered by relevance.
    """
    detected_terms = get_terms(question)
    query_terms = expand_terms(question)
    mappings = []
    seen = set()

    for item in keyword_skill_map:
        if item.get("mapping_status") != "mapped":
            continue

        score = keyword_match_score(
            item.get("keyword") or "",
            question,
            detected_terms,
            query_terms,
        )
        if score == 0:
            continue

        normalized_keyword = normalize_keyword(item.get("keyword"))
        if not is_relevant_mapping_keyword(
            normalized_keyword,
            item.get("source_node_type"),
        ):
            continue
        key = (
            normalized_keyword,
            item.get("source_node_id"),
            item.get("source_node_type"),
            item.get("skill_id"),
        )
        if key in seen:
            continue
        seen.add(key)
        mapping = dict(item)
        mapping["keyword"] = normalized_keyword
        mapping["_relevance_score"] = score + score_text(
            f"{item.get('source_node_title', '')} {item.get('skill', '')}",
            query_terms,
        )
        mappings.append(mapping)

    mappings.sort(key=lambda item: (-item["_relevance_score"], str(item.get("keyword", ""))))
    return mappings, mappings[0]["_relevance_score"] if mappings else 0, query_terms


def belongs_to_skill(node, skill_id, nodes_by_id):
    """Check whether a node belongs to a specific chapter / skill."""
    current = node

    while current:
        if current.get("id") == skill_id:
            return True

        parent_id = current.get("parent_id")
        current = nodes_by_id.get(parent_id)

    return False


def resolve_skill_id(mapping, nodes_by_id):
    """Resolve a mapping to a real skill node in the knowledge-base tree."""
    skill_id = mapping.get("skill_id")
    source_id = mapping.get("source_node_id")
    source_node = nodes_by_id.get(source_id)
    mapped_skill = nodes_by_id.get(skill_id)

    if source_node and mapped_skill and (
        source_node.get("id") == mapped_skill.get("id")
        or belongs_to_skill(source_node, mapped_skill.get("id"), nodes_by_id)
    ):
        return mapped_skill.get("id"), source_node

    if source_node:
        current = source_node
        while current.get("parent_id") and current.get("parent_id") in nodes_by_id:
            current = nodes_by_id[current["parent_id"]]
        if current.get("type") == "chapter":
            return current.get("id"), source_node

    if mapped_skill and mapped_skill.get("type") == "chapter":
        return mapped_skill.get("id"), source_node

    return None, source_node


def rank_learning_objectives(skill_id, nodes, nodes_by_id, query_terms):
    """
    Select the most relevant learning-objective node inside the skill.
    """
    objectives = [
        node for node in nodes
        if node.get("type") == "learning_objective"
        and belongs_to_skill(node, skill_id, nodes_by_id)
    ]

    return sorted(
        objectives,
        key=lambda node: (
            -score_text(f"{node.get('title', '')} {node.get('content', '')}", query_terms),
            node.get("id", ""),
        ),
    )


def rank_learning_activities(skill_id, objective, nodes, nodes_by_id, query_terms):
    """
    Select a relevant practice, activity, or reflection from the same skill.
    """
    allowed_types = {"practice", "activity", "reflection"}

    activities = [
        node for node in nodes
        if node.get("type") in allowed_types
        and belongs_to_skill(node, skill_id, nodes_by_id)
    ]

    return sorted(
        activities,
        key=lambda node: (
            -score_text(f"{node.get('title', '')} {node.get('content', '')}", query_terms),
            node.get("id", ""),
        ),
    )


def node_summary(node):
    if not node:
        return None
    return {
        "id": node.get("id") if node else None,
        "type": node.get("type") if node else None,
        "title": node.get("title") if node else None,
        "content": node.get("content") if node else None,
        "page": node.get("page") if node else None,
    }


def rank_question_relevant_content(skill_id, nodes, nodes_by_id, query_terms):
    """Return the highest-scoring approved content nodes for a skill."""
    relevant_content = []
    for node in nodes:
        if (
            node.get("type") != "content"
            or not belongs_to_skill(node, skill_id, nodes_by_id)
        ):
            continue

        relevance_score = score_text(
            f"{node.get('title', '')} {node.get('content', '')}",
            query_terms,
        )
        if relevance_score > 0:
            relevant_content.append((relevance_score, node))

    relevant_content.sort(key=lambda item: (-item[0], item[1].get("id", "")))
    return [
        node_summary(node)
        for _, node in relevant_content[:QUESTION_RELEVANT_CONTENT_LIMIT]
    ]


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

    mappings, score, query_terms = find_skill_mapping(
        question,
        keyword_skill_map
    )

    nodes_by_id = {
        node["id"]: node
        for node in knowledge_base
        if node.get("id")
    }
    matched_groups = {}
    for mapping in mappings:
        skill_id, source_node = resolve_skill_id(mapping, nodes_by_id)
        if not skill_id:
            continue
        group = matched_groups.setdefault(skill_id, {
            "skill_id": skill_id,
            "skill": mapping.get("skill") or nodes_by_id[skill_id].get("title"),
            "score": 0,
            "matched_keywords": set(),
            "mappings": [],
            "source_nodes": [],
        })
        group["score"] = max(group["score"], mapping["_relevance_score"])
        group["matched_keywords"].add(mapping.get("keyword"))
        group["mappings"].append(mapping)
        source_summary = {
            "id": mapping.get("source_node_id"),
            "type": mapping.get("source_node_type"),
            "title": mapping.get("source_node_title"),
        }
        if source_node:
            source_summary["resolved"] = node_summary(source_node)
        if source_summary not in group["source_nodes"]:
            group["source_nodes"].append(source_summary)

    if not matched_groups:
        return {
            "status": "no_match",
            "student_question": question,
            "message": (
                "I could not find a suitable topic in the approved "
                "SEEK learning content. Please choose an empathy skill "
                "or ask about a related learning topic."
            )
        }

    recommendations = []
    for group in matched_groups.values():
        objectives = rank_learning_objectives(
            group["skill_id"], knowledge_base, nodes_by_id, query_terms
        )
        objective = objectives[0] if objectives else None
        activities = rank_learning_activities(
            group["skill_id"], objective, knowledge_base, nodes_by_id, query_terms
        )
        objective_score = (
            score_text(f"{objective.get('title', '')} {objective.get('content', '')}", query_terms)
            if objective else 0
        )
        activity = activities[0] if activities and (
            score_text(
                f"{activities[0].get('title', '')} {activities[0].get('content', '')}",
                query_terms,
            ) > 0
        ) else None
        activity_reason = (
            "selected_relevant_activity"
            if activity
            else "no_relevant_activity_for_selected_skill_and_question"
        )
        recommendations.append({
            "skill": {"id": group["skill_id"], "title": group["skill"]},
            "relevance_score": (
                group["score"]
                + objective_score
                + (25 if activity else 0)
            ),
            "matched_keywords": sorted(k for k in group["matched_keywords"] if k),
            "source_nodes": group["source_nodes"],
            "mappings": [
                {key: value for key, value in mapping.items() if not key.startswith("_")}
                for mapping in group["mappings"]
            ],
            "learning_objective": node_summary(objective),
            "recommended_activity": node_summary(activity),
            "activity_selection_reason": activity_reason,
            "recommended_learning_items": [
                node_summary(node) for node in activities[:3]
                if score_text(f"{node.get('title', '')} {node.get('content', '')}", query_terms) > 0
            ],
        })

    recommendations.sort(
        key=lambda item: (
            -item["relevance_score"],
            not bool(item["learning_objective"]),
            not bool(item["recommended_activity"]),
            item["skill"]["id"],
        )
    )
    primary = recommendations[0]
    primary_mapping = primary["mappings"][0]
    question_relevant_content = rank_question_relevant_content(
        primary["skill"]["id"],
        knowledge_base,
        nodes_by_id,
        query_terms,
    )

    return {
        "status": "learning_path_selected",
        "student_question": question,
        "detected_emotion": detect_emotion(question),
        "learning_confirmation_required": not requests_actionable_help(question),

        "detected_terms": sorted(query_terms),
        "detected_keywords": sorted(
            get_terms(question).intersection(
                MEANINGFUL_CONCEPTS
                | set(CONCEPT_ALIASES)
            )
        ),
        "matched_keywords": sorted({
            keyword
            for recommendation in recommendations
            for keyword in recommendation["matched_keywords"]
        }),
        "matched_skills": recommendations,
        "recommendations": recommendations,

        "matched_keyword": primary["matched_keywords"][0] if primary["matched_keywords"] else None,

        "skill": primary["skill"],

        "topic": {
            "id": primary_mapping.get("topic_id"),
            "title": primary_mapping.get("topic")
        },

        "learning_objective": primary["learning_objective"],
        "recommended_activity": primary["recommended_activity"],
        "question_relevant_content": question_relevant_content,

        "source_page": primary_mapping.get("page")
    }

def answer_student_question(
    question,
    knowledge_base,
    keyword_skill_map,
    answer_style="short",
):
    if answer_style not in VALID_ANSWER_STYLES:
        raise ValueError("answer_style must be 'short' or 'long'.")

    # 1. Your Pedagogical Controller makes all learning decisions.
    learning_context = process_student_question(
        question,
        knowledge_base,
        keyword_skill_map
    )
    learning_context["answer_style"] = answer_style

    # 2. If the controller could not find an approved lesson,
    #    return its own message. Gemini is not called.
    if learning_context["status"] != "learning_path_selected":
        return learning_context

    # 3. Gemini receives only the controller's selected lesson.
    gemini_response = generate_educational_response(learning_context, answer_style)

    # 4. Return the decision and the four separately-presented learning steps.
    return {
        "learning_context": learning_context,
        "answer": gemini_response["answer"],
        "steps": gemini_response["steps"],
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

import json
import re
import sys
from pathlib import Path

from learning.gemini_responder import generate_educational_response
from learning.semantic_search import semantic_search

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
        if current.get("skill_id") == skill_id:
            return True

        parent_id = current.get("parent_id")
        current = nodes_by_id.get(parent_id)

    return False


def _merge_approved_skill_nodes(knowledge_base):
    """Add objective/content nodes from approved skill definitions when the
    parsed knowledge base is missing the richer curriculum metadata."""
    if not isinstance(knowledge_base, list):
        return knowledge_base

    formatted = [dict(node) for node in knowledge_base if isinstance(node, dict)]
    nodes_by_id = {node.get("id"): node for node in formatted if node.get("id")}

    base_dir = Path(__file__).resolve().parents[2]
    content_skills_path = base_dir / "output" / "content_skills.json"
    if not content_skills_path.exists():
        return formatted

    try:
        with content_skills_path.open("r", encoding="utf-8") as handle:
            skill_data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return formatted

    if not isinstance(skill_data, list):
        return formatted

    chapter_lookup = {
        node.get("skill_id"): node.get("id")
        for node in formatted
        if node.get("type") == "chapter" and node.get("skill_id")
    }
    title_lookup = {
        re.sub(r"^Skill\s*\d+\s*:\s*", "", str(node.get("title") or "")).strip().casefold(): node.get("id")
        for node in formatted
        if node.get("type") == "chapter"
    }

    for skill in skill_data:
        if not isinstance(skill, dict):
            continue
        skill_id = skill.get("skill_id")
        chapter_id = chapter_lookup.get(skill_id)
        if chapter_id is None:
            skill_title = str(skill.get("skill_title") or "").strip()
            chapter_id = title_lookup.get(skill_title.casefold())
        for objective in skill.get("objectives", []) or []:
            if not isinstance(objective, dict):
                continue
            objective_id = objective.get("objective_id")
            if not objective_id:
                continue
            objective_title = (
                objective.get("objective_title")
                or objective.get("title")
                or objective_id
            )
            objective_node = {
                "id": objective_id,
                "skill_id": skill_id,
                "objective_id": objective_id,
                "type": "learning_objective",
                "title": objective_title,
                "content": objective_title,
                "page": None,
            }
            if chapter_id:
                objective_node["parent_id"] = chapter_id
            if objective_id not in nodes_by_id:
                formatted.append(objective_node)
                nodes_by_id[objective_id] = objective_node

            for item in objective.get("learning_content", []) or []:
                if not isinstance(item, dict):
                    continue
                item_id = item.get("item_id")
                if not item_id:
                    continue
                if item_id in nodes_by_id:
                    continue
                content_node = {
                    "id": item_id,
                    "skill_id": skill_id,
                    "objective_id": objective_id,
                    "type": item.get("type") or "content",
                    "title": item.get("title") or item.get("item_id"),
                    "content": item.get("content") or "",
                    "page": item.get("page"),
                }
                if chapter_id:
                    content_node["parent_id"] = objective_id
                formatted.append(content_node)
                nodes_by_id[item_id] = content_node

    return formatted


def curriculum_skill_id(skill_id, nodes, nodes_by_id):
    """Resolve a knowledge-base chapter ID to its saved-progress skill ID."""
    node = nodes_by_id.get(skill_id) or {}
    if node.get("skill_id"):
        return node["skill_id"]

    descendant_skill_ids = {
        descendant.get("skill_id")
        for descendant in nodes
        if descendant.get("skill_id")
        and belongs_to_skill(descendant, skill_id, nodes_by_id)
    }
    if len(descendant_skill_ids) == 1:
        return descendant_skill_ids.pop()
    return skill_id


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


def rank_learning_objectives(
    skill_id,
    nodes,
    nodes_by_id,
    query_terms,
    progress_structure=None,
):
    """
    Select the most relevant learning-objective node inside the skill.
    """
    resolved_curriculum_skill_id = curriculum_skill_id(
        skill_id,
        nodes,
        nodes_by_id,
    )
    chapter = next(
        (
            item
            for item in (progress_structure or {}).get("chapters", [])
            if item.get("chapter_id") == resolved_curriculum_skill_id
        ),
        None,
    )
    if chapter:
        return sorted(
            (
                {
                    "id": objective.get("objective_id"),
                    "objective_id": objective.get("objective_id"),
                    "skill_id": resolved_curriculum_skill_id,
                    "type": "learning_objective",
                    "title": objective.get("title"),
                    "content": objective.get("content"),
                    "page": None,
                }
                for objective in chapter.get("objectives", [])
                if objective.get("objective_id")
            ),
            key=lambda objective: (
                -score_text(
                    f"{objective.get('title', '')} {objective.get('content', '')}",
                    query_terms,
                ),
            ),
        )

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


def rank_learning_activities(
    skill_id,
    objective,
    nodes,
    nodes_by_id,
    query_terms,
    progress_structure=None,
):
    """
    Select a relevant practice, activity, or reflection from the same skill.
    """
    allowed_types = {"practice", "activity", "reflection"}

    knowledge_base_activities = [
        node for node in nodes
        if node.get("type") in allowed_types
        and belongs_to_skill(node, skill_id, nodes_by_id)
    ]
    objective_id = (
        (objective or {}).get("objective_id")
        or (objective or {}).get("id")
    )
    resolved_curriculum_skill_id = (
        (objective or {}).get("skill_id")
        or curriculum_skill_id(skill_id, nodes, nodes_by_id)
    )
    chapter = next(
        (
            item
            for item in (progress_structure or {}).get("chapters", [])
            if item.get("chapter_id") == resolved_curriculum_skill_id
        ),
        None,
    )
    curriculum_objective = next(
        (
            item
            for item in (chapter or {}).get("objectives", [])
            if item.get("objective_id") == objective_id
        ),
        None,
    )
    curriculum_activities = []
    for item in (curriculum_objective or {}).get("items", []):
        item_type = str(item.get("type") or "").lower()
        if item_type not in allowed_types:
            continue
        curriculum_activities.append({
            "id": item.get("item_id"),
            "skill_id": resolved_curriculum_skill_id,
            "objective_id": objective_id,
            "type": item_type,
            "title": item.get("title") or item.get("question"),
            "content": item.get("content") or item.get("instructions") or item.get("question"),
            "page": item.get("page"),
        })

    activities = (
        curriculum_activities
        if progress_structure is not None and curriculum_objective is not None
        else knowledge_base_activities
    )
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
        "skill_id": node.get("skill_id") if node else None,
        "objective_id": node.get("objective_id") if node else None,
        "type": node.get("type") if node else None,
        "title": node.get("title") if node else None,
        "content": node.get("content") if node else None,
        "page": node.get("page") if node else None,
    }


def rank_question_relevant_content(skill_id, nodes, nodes_by_id, query_terms, preferred_objective_id=None):
    """Return the highest-scoring approved content nodes for a skill."""
    relevant_content = []
    query_lower = " ".join(query_terms).casefold()
    for node in nodes:
        if (
            node.get("type") != "content"
            or not belongs_to_skill(node, skill_id, nodes_by_id)
        ):
            continue

        text = f"{node.get('title', '')} {node.get('content', '')}"
        relevance_score = score_text(text, query_terms)
        if preferred_objective_id and node.get("objective_id") == preferred_objective_id:
            relevance_score += 15
        if "spark to forest fire" in text.casefold() or "forest fire" in text.casefold():
            relevance_score += 25
        if "emotional awareness" in text.casefold() and "spark" in query_lower:
            relevance_score += 10
        if relevance_score > 0:
            relevant_content.append((relevance_score, node))

    relevant_content.sort(key=lambda item: (-item[0], item[1].get("id", "")))
    return [
        node_summary(node)
        for _, node in relevant_content[:QUESTION_RELEVANT_CONTENT_LIMIT]
    ]


def _highlight_terms(highlighted_keywords):
    return {
        term
        for item in highlighted_keywords or []
        for term in get_terms(item.get("text", ""))
    }


def _direct_skill_groups(question, knowledge_base, highlighted_keywords=None):
    query_terms = expand_terms(question)
    highlighted_terms = _highlight_terms(highlighted_keywords)
    nodes_by_id = {
        node["id"]: node
        for node in knowledge_base
        if node.get("id")
    }
    chapters = [
        node for node in knowledge_base
        if node.get("type") == "chapter"
    ]
    groups = []

    for chapter in chapters:
        descendants = [
            node for node in knowledge_base
            if belongs_to_skill(node, chapter["id"], nodes_by_id)
        ]
        chapter_text = " ".join(
            f"{node.get('title', '')} {node.get('content', '')}"
            for node in descendants
        )
        score = score_text(chapter_text, query_terms)
        highlighted_overlap = len(highlighted_terms.intersection(query_terms))
        if highlighted_overlap:
            score += highlighted_overlap * 5
        if score <= 0:
            continue

        source_nodes = sorted(
            (
                node for node in descendants
                if node.get("type") in {"content", "learning_objective"}
                and score_text(
                    f"{node.get('title', '')} {node.get('content', '')}",
                    query_terms,
                ) > 0
            ),
            key=lambda node: (
                -score_text(
                    f"{node.get('title', '')} {node.get('content', '')}",
                    query_terms,
                ),
                node.get("id", ""),
            ),
        )
        groups.append({
            "skill_id": chapter["id"],
            "skill": chapter.get("title", ""),
            "score": score,
            "matched_keywords": sorted(
                get_terms(question).intersection(MEANINGFUL_CONCEPTS)
            ),
            "mappings": [],
            "source_nodes": [
                {
                    "id": node["id"],
                    "type": node.get("type"),
                    "title": node.get("title"),
                    "resolved": node_summary(node),
                }
                for node in source_nodes[:3]
            ],
        })

    return groups, query_terms


def _mapped_skill_groups(question, keyword_skill_map, nodes_by_id):
    """Build skill groups only from valid approved keyword mappings."""
    mappings, _, query_terms = find_skill_mapping(question, keyword_skill_map)
    groups_by_skill = {}

    for mapping in mappings:
        skill_id, source_node = resolve_skill_id(mapping, nodes_by_id)
        if not skill_id:
            continue
        group = groups_by_skill.setdefault(skill_id, {
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

    return list(groups_by_skill.values()), query_terms, bool(mappings)


def process_student_question(
    question,
    knowledge_base,
    highlighted_keywords=None,
    progress_structure=None,
):
    """
    Main Pedagogical Controller function.

    Student question
      → identify keywords / skill
      → select learning objective
      → select activity
      → return a structured learning decision
    """
    knowledge_base = _merge_approved_skill_nodes(knowledge_base)
    nodes_by_id = {
        node["id"]: node
        for node in knowledge_base
        if node.get("id")
    }
    has_mapping_records = any(
        isinstance(item, dict) and "mapping_status" in item
        for item in highlighted_keywords or []
    )
    if has_mapping_records:
        matched_groups, query_terms, has_matching_mappings = _mapped_skill_groups(
            question,
            highlighted_keywords,
            nodes_by_id,
        )
        if not has_matching_mappings:
            matched_groups, query_terms = _direct_skill_groups(
                question,
                knowledge_base,
                highlighted_keywords,
            )
    else:
        matched_groups, query_terms = _direct_skill_groups(
            question,
            knowledge_base,
            highlighted_keywords,
        )
        has_matching_mappings = False

    if not matched_groups and not (
        has_mapping_records and has_matching_mappings
    ):
        semantic_results = semantic_search(question, knowledge_base, limit=5)
        if semantic_results:
            primary_skill_id = None
            if semantic_results[0].get("skill_id"):
                primary_skill_id = semantic_results[0]["skill_id"]
                selected_node = nodes_by_id.get(primary_skill_id)
                if not selected_node or selected_node.get("type") != "chapter":
                    chapter = next(
                        (
                            node for node in knowledge_base
                            if node.get("type") == "chapter"
                            and node.get("skill_id") == primary_skill_id
                        ),
                        None,
                    )
                    if chapter:
                        primary_skill_id = chapter["id"]
            if primary_skill_id:
                if not matched_groups:
                    matched_groups = [{
                        "skill_id": primary_skill_id,
                        "skill": next((node.get("title", "") for node in knowledge_base if node.get("id") == primary_skill_id), "Skill"),
                        "score": 1.0,
                        "matched_keywords": sorted(get_terms(question).intersection(MEANINGFUL_CONCEPTS | set(CONCEPT_ALIASES))),
                        "mappings": [],
                        "source_nodes": [{
                            "id": item["id"],
                            "type": item["type"],
                            "title": item["title"],
                            "resolved": {
                                "id": item["id"],
                                "type": item["type"],
                                "title": item["title"],
                                "content": item.get("content"),
                                "page": item.get("page"),
                            },
                        } for item in semantic_results[:3]],
                    }]

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
    for group in matched_groups:
        objectives = rank_learning_objectives(
            group["skill_id"],
            knowledge_base,
            nodes_by_id,
            query_terms,
            progress_structure,
        )
        objective = objectives[0] if objectives else None
        activities = rank_learning_activities(
            group["skill_id"],
            objective,
            knowledge_base,
            nodes_by_id,
            query_terms,
            progress_structure,
        )
        objective_score = (
            score_text(f"{objective.get('title', '')} {objective.get('content', '')}", query_terms)
            if objective else 0
        )
        activity = activities[0] if activities and (
            progress_structure
            or score_text(
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
    question_relevant_content = rank_question_relevant_content(
        primary["skill"]["id"],
        knowledge_base,
        nodes_by_id,
        query_terms,
        preferred_objective_id=(primary.get("learning_objective") or {}).get("objective_id")
        or (primary.get("learning_objective") or {}).get("id"),
    )

    if not question_relevant_content and not has_mapping_records:
        semantic_results = semantic_search(question, knowledge_base, limit=QUESTION_RELEVANT_CONTENT_LIMIT, skill_id=primary["skill"]["id"])
        if semantic_results:
            question_relevant_content = [
                {
                    "id": item["id"],
                    "type": item["type"],
                    "title": item["title"],
                    "content": item["content"],
                    "page": item["page"],
                }
                for item in semantic_results
            ]

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

        "matched_keyword": (
            primary["matched_keywords"][0]
            if primary["matched_keywords"] else None
        ),

        "skill": primary["skill"],

        "topic": {
            "id": None,
            "title": None,
        },

        "learning_objective": primary["learning_objective"],
        "recommended_activity": primary["recommended_activity"],
        "question_relevant_content": question_relevant_content,

        "source_page": (
            primary["source_nodes"][0]["resolved"].get("page")
            if primary["source_nodes"] else None
        )
    }


def _suggestion_target_for_progress(
    learning_context,
    knowledge_base,
    progress_documents,
    progress_structure,
):
    """Select the next saved-progress objective and its approved SEEK content."""
    if (
        not isinstance(progress_documents, list)
        or any(not isinstance(document, dict) for document in progress_documents)
        or not isinstance(progress_structure, dict)
    ):
        return None

    selected_objective = learning_context.get("learning_objective") or {}
    current_objective_id = (
        selected_objective.get("objective_id") or selected_objective.get("id")
    )
    selected_skill = learning_context.get("skill") or {}
    skill_id = selected_skill.get("skill_id") or selected_skill.get("id")
    nodes_by_id = {
        node.get("id"): node
        for node in knowledge_base
        if isinstance(node, dict) and node.get("id")
    }
    skill_node = nodes_by_id.get(skill_id) or {}
    chapter_id = (
        selected_objective.get("skill_id")
        or skill_node.get("skill_id")
        or skill_id
    )
    chapters = progress_structure.get("chapters", [])
    if not isinstance(chapters, list) or any(
        not isinstance(chapter, dict) for chapter in chapters
    ):
        return None
    chapter_index = next(
        (
            index
            for index, chapter in enumerate(chapters)
            if chapter.get("chapter_id") == chapter_id
        ),
        None,
    )
    if chapter_index is None or not current_objective_id:
        return None

    current_progress = next(
        (
            document
            for document in progress_documents
            if document.get("skill_id") == chapter_id
        ),
        None,
    )
    if current_progress is None:
        return None
    completed_current = current_progress.get("completed_objective_ids")
    if not isinstance(completed_current, list):
        return None
    if current_objective_id not in set(completed_current):
        return None

    current_chapter = chapters[chapter_index]
    current_objectives = current_chapter.get("objectives", [])
    if not isinstance(current_objectives, list) or any(
        not isinstance(objective, dict) for objective in current_objectives
    ):
        return None
    objective_index = next(
        (
            index
            for index, objective in enumerate(current_objectives)
            if objective.get("objective_id") == current_objective_id
        ),
        None,
    )
    if objective_index is None:
        return None

    for index in range(chapter_index, len(chapters)):
        chapter = chapters[index]
        progress = next(
            (
                document
                for document in progress_documents
                if document.get("skill_id") == chapter.get("chapter_id")
            ),
            None,
        )
        start = objective_index + 1 if index == chapter_index else 0
        objectives = chapter.get("objectives", [])
        if not isinstance(objectives, list) or any(
            not isinstance(objective, dict) for objective in objectives
        ):
            return None
        future_completed = (
            progress.get("completed_objective_ids", [])
            if progress is not None
            else []
        )
        if not isinstance(future_completed, list):
            return None
        completed_ids = set(future_completed)
        for objective in objectives[start:]:
            objective_id = objective.get("objective_id")
            if not objective_id or objective_id in completed_ids:
                continue
            approved_content = [
                node_summary(node)
                for node in knowledge_base
                if isinstance(node, dict)
                and node.get("type") == "content"
                and node.get("objective_id") == objective_id
                and (
                    node.get("skill_id") == chapter.get("chapter_id")
                    or belongs_to_skill(
                        node,
                        chapter.get("chapter_id"),
                        nodes_by_id,
                    )
                )
            ][:QUESTION_RELEVANT_CONTENT_LIMIT]
            if approved_content:
                return {
                    "skill": {
                        "id": chapter.get("chapter_id"),
                        "skill_id": chapter.get("chapter_id"),
                        "title": chapter.get("title"),
                    },
                    "learning_objective": {
                        "id": objective_id,
                        "objective_id": objective_id,
                        "skill_id": chapter.get("chapter_id"),
                        "type": "learning_objective",
                        "title": objective.get("title"),
                        "content": objective.get("content"),
                    },
                    "recommended_activity": next(
                        (
                            {
                                "id": item.get("item_id"),
                                "skill_id": chapter.get("chapter_id"),
                                "objective_id": objective_id,
                                "type": item.get("type"),
                                "title": item.get("title") or item.get("question"),
                                "content": (
                                    item.get("content")
                                    or item.get("instructions")
                                    or item.get("question")
                                ),
                                "page": item.get("page"),
                            }
                            for item in objective.get("items", [])
                            if item.get("type") in {"activity", "practice", "reflection"}
                            and item.get("item_id")
                        ),
                        None,
                    ),
                    "question_relevant_content": approved_content,
                }
            return None
        objective_index = -1

    return None


def answer_student_question(
    question,
    knowledge_base,
    highlighted_keywords=None,
    answer_style="short",
    progress_documents=None,
    progress_structure=None,
):
    if answer_style not in VALID_ANSWER_STYLES:
        raise ValueError("answer_style must be 'short' or 'long'.")

    # 1. Your Pedagogical Controller makes all learning decisions.
    learning_context = process_student_question(
        question,
        knowledge_base,
        highlighted_keywords,
        progress_structure,
    )
    learning_context["answer_style"] = answer_style

    # 2. If the controller could not find an approved lesson,
    #    return its own message. Gemini is not called.
    if learning_context["status"] != "learning_path_selected":
        learning_context["suggested_questions"] = []
        return learning_context

    suggestion_target = _suggestion_target_for_progress(
        learning_context,
        knowledge_base,
        progress_documents,
        progress_structure,
    )
    responder_context = learning_context
    if suggestion_target:
        learning_context["skill"] = suggestion_target["skill"]
        learning_context["learning_objective"] = suggestion_target[
            "learning_objective"
        ]
        learning_context["recommended_activity"] = suggestion_target[
            "recommended_activity"
        ]
        learning_context["question_relevant_content"] = suggestion_target[
            "question_relevant_content"
        ]
        responder_context = {
            **learning_context,
            "suggestion_learning_objective": suggestion_target[
                "learning_objective"
            ],
            "suggestion_relevant_content": suggestion_target[
                "question_relevant_content"
            ],
        }

    # 3. Gemini receives the selected lesson and, when completed, the next
    #    incomplete objective for suggestions in the same request.
    gemini_response = generate_educational_response(responder_context, answer_style)

    # 4. Return the decision and the four separately-presented learning steps.
    return {
        "learning_context": learning_context,
        "answer": gemini_response["answer"],
        "steps": gemini_response["steps"],
        "suggested_questions": gemini_response.get("suggested_questions", []),
    }


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]

    knowledge_base = load_json(
        BASE_DIR / "output" / "knowledge_base.json"
    )

    highlighted_keywords = load_json(
        BASE_DIR / "output" / "highlighted_keywords.json"
    )

    question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "I feel overwhelmed before exams. How can I calm down?"
    )

    result = answer_student_question(
        question,
        knowledge_base,
        highlighted_keywords
    )

    print(json.dumps(result, indent=4, ensure_ascii=False))

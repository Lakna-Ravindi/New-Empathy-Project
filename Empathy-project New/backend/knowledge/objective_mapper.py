"""Conservative mapping of extracted knowledge nodes to authored objectives."""

from __future__ import annotations

import re
from typing import Any, Iterable


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "the", "their", "this",
    "to", "we", "what", "when", "with", "you", "your", "will", "able",
    "apply", "define", "describe", "discuss", "explain", "identify", "name",
    "recognize", "use", "understand", "learn", "develop", "practice",
}

NON_CONTENT_TYPES = {"chapter", "module", "topic", "objective_heading"}


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if token not in STOP_WORDS and len(token) > 2
    }


def _node_text(node: dict[str, Any]) -> str:
    return " ".join(
        str(node.get(field, ""))
        for field in ("title", "content")
        if node.get(field)
    )


def _flatten_objectives(content_skills: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    objectives = []
    for skill in content_skills:
        for objective in skill.get("objectives", []):
            objectives.append({
                "skill_id": skill.get("skill_id"),
                "skill_title": skill.get("skill_title", ""),
                "key_concepts": skill.get("key_concepts", []),
                **objective,
            })
    return objectives


def _skill_id_for_node(node, nodes_by_id, skills):
    current = node
    visited = set()
    while current and current.get("id") not in visited:
        visited.add(current.get("id"))
        if current.get("type") == "chapter":
            title = current.get("title", "")
            number = re.search(r"skill\s+(\d+)", title, re.IGNORECASE)
            if number:
                candidate = f"skill_{int(number.group(1)):02d}"
                if any(skill.get("skill_id") == candidate for skill in skills):
                    return candidate
            title_tokens = _tokens(title)
            for skill in skills:
                if title_tokens & _tokens(skill.get("skill_title", "")):
                    return skill.get("skill_id")
            return None
        current = nodes_by_id.get(current.get("parent_id"))
    return None


def _score(node, objective):
    node_tokens = _tokens(_node_text(node))
    title_tokens = _tokens(objective.get("objective_title", ""))
    concept_tokens = _tokens(" ".join(objective.get("key_concepts", [])))
    title_overlap = len(node_tokens & title_tokens)
    concept_overlap = len(node_tokens & concept_tokens)
    return title_overlap * 3 + concept_overlap, title_overlap, concept_overlap


def map_nodes_to_objectives(nodes, content_skills):
    """Annotate nodes without forcing low-confidence objective assignments."""
    skills = list(content_skills)
    objectives = _flatten_objectives(skills)
    nodes_by_id = {node.get("id"): node for node in nodes}
    objectives_by_id = {
        objective.get("objective_id"): objective
        for objective in objectives
        if objective.get("objective_id")
    }

    for node in nodes:
        node["skill_id"] = None
        node["objective_id"] = None
        node["match_status"] = "unassigned"

        if node.get("type") in NON_CONTENT_TYPES:
            node["match_status"] = "not_applicable"
            continue

        skill_id = _skill_id_for_node(node, nodes_by_id, skills)
        candidates = [
            objective for objective in objectives
            if not skill_id or objective.get("skill_id") == skill_id
        ]
        scored = sorted(
            ((_score(node, objective), objective) for objective in candidates),
            key=lambda entry: (entry[0][0], entry[0][1], entry[1].get("objective_id", "")),
            reverse=True,
        )
        if not scored:
            continue

        best_score, best = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else (0, 0, 0)
        score, title_overlap, _ = best_score
        node["skill_id"] = skill_id or best.get("skill_id")
        if score >= 3 and title_overlap >= 1 and score > second_score[0]:
            node["objective_id"] = best.get("objective_id")
            node["skill_id"] = best.get("skill_id")
            node["match_status"] = "matched"
        elif score > 0:
            node["match_status"] = "review"

    return nodes, objectives_by_id

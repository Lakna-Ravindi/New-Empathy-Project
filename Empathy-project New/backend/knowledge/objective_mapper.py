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



def _pdf_objective_number(node):
    """Extract objective number from explicit or numbered PDF objectives."""
    text = str(node.get("title", "") or node.get("content", ""))
    match = re.search(r"\bobjective\s+(\d+)\b", text, re.IGNORECASE)
    if not match:
        match = re.match(r"\s*(\d+)[.)]\s+", text)
    return int(match.group(1)) if match else None


def _is_explicit_pdf_objective(node):
    text = str(node.get("title", "") or node.get("content", ""))
    return bool(re.search(r"\bobjective\s+\d+\b", text, re.IGNORECASE))


def _objective_id_for_number(skill_id: str | None, number: int | None) -> str | None:
    canonical = {
        "skill_01": {1: "obj_01_01", 2: "obj_01_05", 3: "obj_01_02",
                     4: "obj_01_06", 5: "obj_01_03", 6: "obj_01_04"},
        "skill_02": {1: "obj_02_01", 2: "obj_02_02", 3: "obj_02_03",
                     4: "obj_02_04", 5: "obj_02_05"},
        "skill_03": {1: "obj_03_01", 2: "obj_03_05", 3: "obj_03_06",
                     4: "obj_03_02", 5: "obj_03_03", 6: "obj_03_04"},
        "skill_04": {1: "obj_04_01", 2: "obj_04_02", 3: "obj_04_05",
                     4: "obj_04_06", 5: "obj_04_07", 6: "obj_04_03",
                     7: "obj_04_04"},
    }
    return canonical.get(skill_id, {}).get(number)


def _nearest_objective(node, nodes_by_id):
    current = node
    visited = set()
    while current and current.get("id") not in visited:
        visited.add(current.get("id"))
        if current.get("type") == "learning_objective":
            return current
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

        skill_id = _skill_id_for_node(node, nodes_by_id, skills)

        if node.get("type") == "chapter":
            node["skill_id"] = skill_id
            node["match_status"] = "not_applicable"
            continue

        if node.get("type") in NON_CONTENT_TYPES:
            node["skill_id"] = skill_id
            node["match_status"] = "not_applicable"
            continue

        if node.get("type") == "learning_objective":
            heading_number = _pdf_objective_number(node)
            target_id = _objective_id_for_number(skill_id, heading_number)
            if target_id:
                node["skill_id"] = skill_id
                node["objective_id"] = target_id
                node["match_status"] = "matched"
            else:
                node["skill_id"] = skill_id
                node["match_status"] = "review"
            continue

        parent_objective = _nearest_objective(node, nodes_by_id)
        if parent_objective:
            parent_skill = _skill_id_for_node(parent_objective, nodes_by_id, skills)
            objective_number = _pdf_objective_number(parent_objective)
            objective_id = _objective_id_for_number(parent_skill, objective_number)
            if objective_id:
                node["skill_id"] = parent_skill
                node["objective_id"] = objective_id
                node["match_status"] = "matched"
                continue

        if skill_id:
            node["skill_id"] = skill_id
            node["match_status"] = "review"
            continue

    return nodes, objectives_by_id

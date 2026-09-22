"""Build a skill-wise progress blueprint from the generated knowledge base."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


ACTIONABLE_TYPES = {"activity", "practice", "assessment", "reflection"}
DEFAULT_KB_PATH = Path(__file__).resolve().parents[2] / "output" / "knowledge_base.json"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[2] / "output" / "progress_structure.json"

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "the", "their", "this",
    "to", "we", "what", "when", "with", "you", "your",
}


def _text(node: dict[str, Any]) -> str:
    return " ".join(
        str(node.get(field, ""))
        for field in ("title", "content")
        if node.get(field)
    )


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if token not in STOP_WORDS and len(token) > 2
    }


def _ancestor_ids(node: dict[str, Any], nodes_by_id: dict[str, dict[str, Any]]) -> list[str]:
    ancestors: list[str] = []
    current_id = node.get("parent_id")
    visited: set[str] = set()

    while current_id and current_id not in visited:
        visited.add(current_id)
        parent = nodes_by_id.get(current_id)
        if parent is None:
            break
        ancestors.append(current_id)
        current_id = parent.get("parent_id")

    return ancestors


def _chapter_for(node: dict[str, Any], nodes_by_id: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if node.get("type") == "chapter":
        return node

    for ancestor_id in _ancestor_ids(node, nodes_by_id):
        ancestor = nodes_by_id[ancestor_id]
        if ancestor.get("type") == "chapter":
            return ancestor

    return None


def _topic_ids(node: dict[str, Any], nodes_by_id: dict[str, dict[str, Any]]) -> list[str]:
    return [
        ancestor_id
        for ancestor_id in _ancestor_ids(node, nodes_by_id)
        if nodes_by_id[ancestor_id].get("type") == "topic"
    ]


def _best_objective(
    item: dict[str, Any],
    objectives: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, int]:
    item_tokens = _tokens(_text(item))
    if not item_tokens:
        return None, 0

    scored = []
    for objective in objectives:
        overlap = len(item_tokens & _tokens(_text(objective)))
        scored.append((overlap, objective.get("page", 0), objective.get("id", ""), objective))

    best_overlap, _, _, best = max(scored, key=lambda entry: (entry[0], -entry[1], entry[2]))
    if best_overlap == 0:
        return None, 0
    return best, best_overlap


def build_progress_structure(nodes: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return a progress blueprint without changing the knowledge-base nodes."""
    node_list = list(nodes)
    nodes_by_id = {node["id"]: node for node in node_list if node.get("id")}
    chapters = [node for node in node_list if node.get("type") == "chapter"]
    actionable = [node for node in node_list if node.get("type") in ACTIONABLE_TYPES]

    structure: list[dict[str, Any]] = []
    chapter_by_id: dict[str, dict[str, Any]] = {}

    for chapter in chapters:
        chapter_entry = {
            "chapter_id": chapter["id"],
            "title": chapter.get("title", ""),
            "page": chapter.get("page"),
            "objectives": [],
            "unassigned_items": [],
        }
        structure.append(chapter_entry)
        chapter_by_id[chapter["id"]] = chapter_entry

        objectives = [
            node
            for node in node_list
            if node.get("type") == "learning_objective"
            and _chapter_for(node, nodes_by_id)
            and _chapter_for(node, nodes_by_id).get("id") == chapter["id"]
        ]
        for objective in objectives:
            chapter_entry["objectives"].append(
                {
                    "objective_id": objective["id"],
                    "title": objective.get("title", ""),
                    "content": objective.get("content", ""),
                    "page": objective.get("page"),
                    "items": [],
                }
            )

    objective_entries = {
        objective["objective_id"]: objective
        for chapter in structure
        for objective in chapter["objectives"]
    }

    for item in actionable:
        chapter = _chapter_for(item, nodes_by_id)
        if chapter is None or chapter["id"] not in chapter_by_id:
            continue

        item_entry = {
            "item_id": item["id"],
            "type": item.get("type"),
            "title": item.get("title", ""),
            "page": item.get("page"),
            "topic_ids": _topic_ids(item, nodes_by_id),
            "source_parent_id": item.get("parent_id"),
        }
        chapter_entry = chapter_by_id[chapter["id"]]
        objective_nodes = [
            node
            for node in node_list
            if node.get("type") == "learning_objective"
            and _chapter_for(node, nodes_by_id)
            and _chapter_for(node, nodes_by_id).get("id") == chapter["id"]
        ]
        objective, overlap = _best_objective(item, objective_nodes)
        if objective is not None:
            objective_entries[objective["id"]]["items"].append(
                {**item_entry, "objective_match_tokens": overlap}
            )
        else:
            chapter_entry["unassigned_items"].append(item_entry)

    return {
        "version": 1,
        "source": "output/knowledge_base.json",
        "item_types": sorted(ACTIONABLE_TYPES),
        "chapters": structure,
        "summary": {
            "chapter_count": len(structure),
            "objective_count": sum(len(chapter["objectives"]) for chapter in structure),
            "actionable_item_count": len(actionable),
            "assigned_item_count": sum(
                len(objective["items"])
                for chapter in structure
                for objective in chapter["objectives"]
            ),
            "unassigned_item_count": sum(
                len(chapter["unassigned_items"]) for chapter in structure
            ),
        },
    }


def load_nodes(path: Path = DEFAULT_KB_PATH) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Knowledge base must contain a JSON list: {path}")
    return data


def write_progress_structure(
    input_path: Path = DEFAULT_KB_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    structure = build_progress_structure(load_nodes(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(structure, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return structure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_KB_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    structure = write_progress_structure(args.input, args.output)
    print(json.dumps(structure["summary"], indent=2))
    print(f"Wrote progress structure to {args.output}")


if __name__ == "__main__":
    main()

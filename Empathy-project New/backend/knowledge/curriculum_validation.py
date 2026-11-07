"""Validate and reconcile generated curriculum identifiers without inventing content."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge.objective_mapper import CANONICAL_OBJECTIVE_IDS


def _load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_curriculum(output_dir: Path) -> dict[str, Any]:
    knowledge_base = _load(output_dir / "knowledge_base.json")
    progress = _load(output_dir / "progress_structure.json")
    content_skills = _load(output_dir / "content_skills.json")
    keyword_map = _load(output_dir / "keyword_skill_map.json")

    canonical_skills = set(CANONICAL_OBJECTIVE_IDS)
    canonical_objectives = {
        objective_id
        for objectives in CANONICAL_OBJECTIVE_IDS.values()
        for objective_id in objectives.values()
    }
    node_by_id = {node.get("id"): node for node in knowledge_base}
    progress_pairs = {
        (chapter.get("chapter_id"), objective.get("objective_id"))
        for chapter in progress.get("chapters", [])
        for objective in chapter.get("objectives", [])
    }
    invalid_nodes = [
        node.get("id")
        for node in knowledge_base
        if node.get("skill_id") not in canonical_skills
        or (
            node.get("objective_id") is not None
            and node.get("objective_id") not in canonical_objectives
        )
    ]
    invalid_progress = sorted(
        [
            [skill_id, objective_id]
            for skill_id, objective_id in progress_pairs
            if skill_id not in canonical_skills
            or objective_id not in canonical_objectives
        ]
    )
    extra_authored_skills = sorted(
        skill.get("skill_id")
        for skill in content_skills
        if skill.get("skill_id") not in canonical_skills
    )
    legacy_keyword_ids = sorted({
        entry.get("skill_id")
        for entry in keyword_map
        if entry.get("skill_id") not in canonical_skills
    })
    actionable = {
        objective.get("objective_id"): len(objective.get("items", []))
        for chapter in progress.get("chapters", [])
        for objective in chapter.get("objectives", [])
    }
    objective_validation = build_objective_validation(
        knowledge_base,
        progress,
        canonical_skills,
    )
    progress_item_ids = [
        item.get("item_id")
        for chapter in progress.get("chapters", [])
        for objective in chapter.get("objectives", [])
        for item in objective.get("items", [])
    ]
    referenced_pairs = [
        (chapter.get("chapter_id"), objective.get("objective_id"))
        for chapter in progress.get("chapters", [])
        for objective in chapter.get("objectives", [])
        for item in objective.get("items", [])
    ]
    duplicate_objective_ids = len(progress_pairs) != sum(
        len(chapter.get("objectives", []))
        for chapter in progress.get("chapters", [])
    )
    duplicate_item_ids = len(progress_item_ids) != len(set(progress_item_ids))
    invalid_item_references = sorted({
        f"{skill_id}:{objective_id}"
        for skill_id, objective_id in referenced_pairs
        if (skill_id, objective_id) not in progress_pairs
    })

    return {
        "canonical_skill_ids": sorted(canonical_skills),
        "canonical_objective_ids": sorted(canonical_objectives),
        "knowledge_base": {
            "node_count": len(knowledge_base),
            "duplicate_node_ids": len(node_by_id) != len(knowledge_base),
            "invalid_node_ids": invalid_nodes,
        },
        "progress_structure": {
            "objective_count": len(progress_pairs),
            "invalid_objectives": invalid_progress,
            "actionable_item_counts": actionable,
            "objectives_without_actionable_items": sorted(
                objective_id for objective_id, count in actionable.items() if count == 0
            ),
            "duplicate_objective_ids": duplicate_objective_ids,
            "duplicate_item_ids": duplicate_item_ids,
            "invalid_item_references": invalid_item_references,
        },
        "content_skills": {
            "skill_count": len(content_skills),
            "out_of_scope_skill_ids": extra_authored_skills,
        },
        "keyword_skill_map": {
            "entry_count": len(keyword_map),
            "legacy_skill_ids": legacy_keyword_ids,
        },
        "objective_validation": objective_validation,
        "status": "valid"
        if (
            not invalid_nodes
            and not invalid_progress
            and not duplicate_objective_ids
            and not duplicate_item_ids
            and not invalid_item_references
        )
        else "requires_review",
        "notes": [
            "Out-of-scope authored skills were not merged into the canonical SEEK progress structure.",
            "Objectives without actionable items remain unresolved pending approved SEEK source content.",
        ],
    }


def build_objective_validation(
    knowledge_base: list[dict[str, Any]],
    progress: dict[str, Any],
    canonical_skills: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return one source-to-progress diagnostic for every canonical objective."""
    canonical_skills = canonical_skills or set(CANONICAL_OBJECTIVE_IDS)
    report = []
    for chapter in progress.get("chapters", []):
        skill_id = chapter.get("chapter_id")
        if skill_id not in canonical_skills:
            continue
        for objective in chapter.get("objectives", []):
            objective_id = objective.get("objective_id")
            nodes = [
                node for node in knowledge_base
                if node.get("skill_id") == skill_id
                and node.get("objective_id") == objective_id
            ]
            item_count = len(objective.get("items", []))
            if item_count:
                missing_status = "supported"
                correction = "No correction required."
            elif nodes:
                missing_status = "approved_content_without_actionable_item"
                correction = (
                    "Review SEEK extraction and item classification; do not "
                    "invent an item."
                )
            else:
                missing_status = "no_linked_approved_content"
                correction = (
                    "Confirm the approved SEEK mapping; do not create content "
                    "without source evidence."
                )
            report.append({
                "skill_id": skill_id,
                "objective_id": objective_id,
                "objective_title": objective.get("title", ""),
                "related_knowledge_base_nodes": len(nodes),
                "actionable_progress_items": item_count,
                "mapping_status": (
                    "matched" if nodes else "unmapped"
                ),
                "missing_content_status": missing_status,
                "recommended_correction": correction,
            })
    return report


def reconcile_keyword_skill_ids(output_dir: Path) -> int:
    """Rewrite only legacy keyword IDs when their source node has an approved mapping."""
    kb = _load(output_dir / "knowledge_base.json")
    keyword_path = output_dir / "keyword_skill_map.json"
    keyword_map = _load(keyword_path)
    changed = 0
    for entry in keyword_map:
        source = next(
            (node for node in kb if node.get("id") == entry.get("source_node_id")),
            None,
        )
        if not source or source.get("skill_id") not in CANONICAL_OBJECTIVE_IDS:
            continue
        if entry.get("skill_id") != source["skill_id"]:
            entry["skill_id"] = source["skill_id"]
            changed += 1
        if source.get("objective_id") and entry.get("objective_id") != source["objective_id"]:
            entry["objective_id"] = source["objective_id"]
            changed += 1
    keyword_path.write_text(json.dumps(keyword_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return changed


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    changed = reconcile_keyword_skill_ids(root / "output")
    report = validate_curriculum(root / "output")
    report["keyword_skill_map"]["reconciled_fields"] = changed
    (root / "output" / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))

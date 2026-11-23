"""Curriculum tools for retrieving approved UNESCO MGIEP SEEK content and activities.

Guarantees:
- Reuses existing semantic_search and knowledge base artifacts.
- Only retrieves approved UNESCO SEEK content.
- Prevents LLM/agent hallucination of non-canonical skills, objectives, or activities.
- Read-only: does not modify any student progress.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.state import (
    ALL_CANONICAL_OBJECTIVE_IDS,
    ALL_CANONICAL_SKILL_IDS,
)
from agents.tools.registry import ToolValidationError
from learning.semantic_search import semantic_search

logger = logging.getLogger(__name__)

# Canonical mapping for supplementary curriculum objectives to their authored parent objectives
OBJECTIVE_CANONICAL_MAP: Dict[str, str] = {
    "obj_01_05": "obj_01_01",  # Sensations vocabulary -> Body literacy
    "obj_01_06": "obj_01_02",  # Recognize out of resilience zone -> Resilient zone
    "obj_02_05": "obj_02_01",  # Ethical mindfulness scenarios -> Ethical mindfulness
    "obj_03_05": "obj_03_01",  # Emotional awareness importance -> Emotional awareness
    "obj_03_06": "obj_03_02",  # Afflictive & beneficial states -> Nature of thoughts
    "obj_04_05": "obj_04_01",  # Self compassion importance -> Self compassion
    "obj_04_06": "obj_04_02",  # Practice self compassion -> Thinking traps & self compassion
    "obj_04_07": "obj_04_01",  # Cultivate compassion steps -> Self compassion
}

# Base output path
_BASE_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "output"

_CACHED_KNOWLEDGE_BASE: Optional[List[Dict[str, Any]]] = None
_CACHED_CONTENT_SKILLS: Optional[List[Dict[str, Any]]] = None
_CACHED_PROGRESS_STRUCTURE: Optional[Dict[str, Any]] = None


def get_knowledge_base() -> List[Dict[str, Any]]:
    """Loads and caches the authoritative UNESCO SEEK knowledge base."""
    global _CACHED_KNOWLEDGE_BASE
    if _CACHED_KNOWLEDGE_BASE is None:
        kb_path = _BASE_OUTPUT_DIR / "knowledge_base.json"
        if not kb_path.exists():
            raise FileNotFoundError(f"Authoritative knowledge base not found at: {kb_path}")
        with kb_path.open("r", encoding="utf-8") as file:
            _CACHED_KNOWLEDGE_BASE = json.load(file)
    return _CACHED_KNOWLEDGE_BASE


def get_content_skills() -> List[Dict[str, Any]]:
    """Loads and caches the approved SEEK content skills."""
    global _CACHED_CONTENT_SKILLS
    if _CACHED_CONTENT_SKILLS is None:
        skills_path = _BASE_OUTPUT_DIR / "content_skills.json"
        if not skills_path.exists():
            raise FileNotFoundError(f"Content skills file not found at: {skills_path}")
        with skills_path.open("r", encoding="utf-8") as file:
            _CACHED_CONTENT_SKILLS = json.load(file)
    return _CACHED_CONTENT_SKILLS


def get_progress_structure() -> Dict[str, Any]:
    """Loads and caches the canonical progress structure tree."""
    global _CACHED_PROGRESS_STRUCTURE
    if _CACHED_PROGRESS_STRUCTURE is None:
        try:
            from learning.progress_service import ProgressService
            _CACHED_PROGRESS_STRUCTURE = ProgressService().get_structure()
        except Exception:
            prog_path = _BASE_OUTPUT_DIR / "progress_structure.json"
            if not prog_path.exists():
                raise FileNotFoundError(f"Progress structure file not found at: {prog_path}")
            with prog_path.open("r", encoding="utf-8") as file:
                _CACHED_PROGRESS_STRUCTURE = json.load(file)
    return _CACHED_PROGRESS_STRUCTURE


_CACHED_KEYWORD_SKILL_MAP: Optional[List[Dict[str, Any]]] = None


def get_keyword_skill_map() -> List[Dict[str, Any]]:
    """Loads and caches the keyword-to-skill mapping."""
    global _CACHED_KEYWORD_SKILL_MAP
    if _CACHED_KEYWORD_SKILL_MAP is None:
        map_path = _BASE_OUTPUT_DIR / "keyword_skill_map.json"
        if not map_path.exists():
            return []
        with map_path.open("r", encoding="utf-8") as file:
            _CACHED_KEYWORD_SKILL_MAP = json.load(file)
    return _CACHED_KEYWORD_SKILL_MAP


def retrieve_approved_seek_content(
    query: str,
    skill_id: Optional[str] = None,
    objective_id: Optional[str] = None,
    limit: int = 5,
    knowledge_base: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Tool 1: Retrieves approved UNESCO SEEK curriculum content chunks.

    Reuses existing semantic_search and authoritative knowledge_base.json.
    Strictly read-only and restricted to approved UNESCO SEEK nodes.
    When strict objective filtering produces zero results, retries retrieval within
    the same approved skill without the objective restriction.
    """
    if not query or not isinstance(query, str) or not query.strip():
        raise ToolValidationError("query must be a non-empty string.")

    cleaned_query = query.strip()

    if skill_id is not None:
        normalized_skill = skill_id.strip()
        if normalized_skill not in ALL_CANONICAL_SKILL_IDS:
            raise ToolValidationError(
                f"Invalid skill_id '{skill_id}'. Must be an approved SEEK skill (skill_01 to skill_08)."
            )
        skill_id = normalized_skill

    if objective_id is not None:
        normalized_obj = objective_id.strip()
        if normalized_obj not in ALL_CANONICAL_OBJECTIVE_IDS:
            raise ToolValidationError(
                f"Invalid objective_id '{objective_id}'. Must be a canonical SEEK objective."
            )
        objective_id = normalized_obj

    if not isinstance(limit, int) or limit < 1 or limit > 20:
        raise ToolValidationError("limit must be an integer between 1 and 20.")

    kb = knowledge_base if knowledge_base is not None else get_knowledge_base()

    effective_skill_id = skill_id
    if not effective_skill_id and objective_id:
        parts = objective_id.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            cand_skill = f"skill_{parts[1]}"
            if cand_skill in ALL_CANONICAL_SKILL_IDS:
                effective_skill_id = cand_skill

    search_limit = max(limit * 3, 15) if objective_id else limit
    results = semantic_search(cleaned_query, kb, limit=search_limit, skill_id=skill_id or effective_skill_id)

    # Filter and format approved content chunks
    approved_chunks = []
    for item in results:
        # If objective_id was specified, filter by objective
        if objective_id and item.get("objective_id") and item.get("objective_id") != objective_id:
            continue

        chunk = {
            "id": item.get("id"),
            "title": item.get("title", ""),
            "content": item.get("content", ""),
            "page": item.get("page"),
            "skill_id": item.get("skill_id") or effective_skill_id or skill_id,
            "objective_id": item.get("objective_id") or objective_id,
            "score": round(float(item.get("score", 0.0)), 4),
            "source": item.get("source", "SEEK_Learning.pdf"),
        }
        approved_chunks.append(chunk)
        if len(approved_chunks) >= limit:
            break

    # When strict objective filtering produces zero results, retry retrieval within the same approved skill without the objective restriction
    if not approved_chunks and objective_id:
        logger.info(
            "Strict objective filtering for '%s' produced zero results. "
            "Retrying retrieval within approved skill '%s' without objective restriction.",
            objective_id,
            effective_skill_id or skill_id,
        )
        fallback_results = (
            results if (skill_id or effective_skill_id)
            else semantic_search(cleaned_query, kb, limit=limit, skill_id=effective_skill_id)
        )
        for item in fallback_results:
            if effective_skill_id and item.get("skill_id") and item.get("skill_id") != effective_skill_id:
                continue
            chunk = {
                "id": item.get("id"),
                "title": item.get("title", ""),
                "content": item.get("content", ""),
                "page": item.get("page"),
                "skill_id": item.get("skill_id") or effective_skill_id or skill_id,
                "objective_id": item.get("objective_id") or objective_id,
                "score": round(float(item.get("score", 0.0)), 4),
                "source": item.get("source", "SEEK_Learning.pdf"),
            }
            approved_chunks.append(chunk)
            if len(approved_chunks) >= limit:
                break

    return approved_chunks


def get_learning_activity(
    skill_id: str,
    objective_id: Optional[str] = None,
    activity_id: Optional[str] = None,
    progress_structure: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Tool 5: Retrieves an approved SEEK learning activity.

    Reuses existing progress_structure.json and content_skills.json.
    Prevents agents from inventing unapproved activities. Strictly read-only.
    """
    if not skill_id or not isinstance(skill_id, str):
        raise ToolValidationError("skill_id must be a non-empty string.")

    normalized_skill = skill_id.strip()
    if normalized_skill not in ALL_CANONICAL_SKILL_IDS:
        raise ToolValidationError(
            f"Invalid skill_id '{skill_id}'. Must be a canonical SEEK skill."
        )

    if objective_id is not None:
        normalized_obj = objective_id.strip()
        if normalized_obj not in ALL_CANONICAL_OBJECTIVE_IDS:
            raise ToolValidationError(
                f"Invalid objective_id '{objective_id}'. Must be a canonical SEEK objective."
            )
        objective_id = normalized_obj

    structure = progress_structure if progress_structure is not None else get_progress_structure()
    chapters = structure.get("chapters", [])

    chapter = next((c for c in chapters if c.get("chapter_id") == normalized_skill), None)
    if not chapter:
        raise ToolValidationError(f"Skill chapter '{normalized_skill}' not found in curriculum structure.")

    matched_item = None
    parent_objective = None

    # Search for specific activity_id or first activity in objective / chapter
    for obj in chapter.get("objectives", []):
        obj_id = obj.get("objective_id")
        if objective_id and obj_id != objective_id:
            continue

        for item in obj.get("items", []):
            item_type = str(item.get("type", "")).lower()
            if item_type in {"activity", "practice", "reflection"}:
                if activity_id:
                    if item.get("item_id") == activity_id.strip():
                        matched_item = item
                        parent_objective = obj
                        break
                else:
                    matched_item = item
                    parent_objective = obj
                    break
        if matched_item:
            break

    # If no activity was found for a supplementary objective, check its canonical mapped objective
    if not matched_item and objective_id and not activity_id:
        fallback_obj_id = OBJECTIVE_CANONICAL_MAP.get(objective_id)
        if fallback_obj_id:
            for obj in chapter.get("objectives", []):
                if obj.get("objective_id") != fallback_obj_id:
                    continue
                for item in obj.get("items", []):
                    item_type = str(item.get("type", "")).lower()
                    if item_type in {"activity", "practice", "reflection"}:
                        matched_item = item
                        parent_objective = obj
                        break
                if matched_item:
                    break

    if not matched_item:
        if activity_id:
            raise ToolValidationError(
                f"Activity '{activity_id}' is not an approved activity in {normalized_skill}."
            )
        raise ToolValidationError(
            f"No approved activities found for skill '{normalized_skill}' "
            f"{'and objective ' + objective_id if objective_id else ''}."
        )

    # Return normalized activity contract
    interaction_type = matched_item.get("interaction_type")
    if interaction_type not in {"practical", "reflective", "combined"}:
        interaction_type = "reflective" if matched_item.get("type") == "reflection" else "practical"

    result = {
        "id": matched_item.get("item_id"),
        "progress_item_id": matched_item.get("item_id"),
        "progress_item_type": "activity",
        "title": matched_item.get("title", ""),
        "content": matched_item.get("content", ""),
        "type": matched_item.get("type", "activity"),
        "interaction_type": interaction_type,
        "skill_id": normalized_skill,
        "objective_id": (parent_objective or {}).get("objective_id"),
    }
    for field in ("scenario", "instructions", "options", "best_answer", "feedback", "duration"):
        if matched_item.get(field) is not None:
            result[field] = matched_item[field]

    return result

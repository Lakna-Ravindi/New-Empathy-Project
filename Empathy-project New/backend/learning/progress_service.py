"""Calculate learning progress from the assigned curriculum structure."""

import json
from pathlib import Path

from learning.chatbot_practice import CHATBOT_PRACTICES
from knowledge.objective_mapper import CANONICAL_OBJECTIVE_IDS


def activity_metadata(activity):
    """Return normalized metadata for a trackable activity."""
    interaction_type = activity.get("interaction_type")
    if interaction_type not in {"practical", "reflective", "combined"}:
        interaction_type = (
            "reflective" if activity.get("type") == "reflection" else "practical"
        )
    return {"interaction_type": interaction_type}


class ProgressService:
    """Build objective, skill, and overall progress summaries."""

    DEFAULT_QUIZ_PASSING_SCORE = 90
    ACTIVITY_ELIGIBILITY_PERCENT = 50

    def __init__(self, structure_path=None, structure=None):
        if structure is not None:
            self.structure = structure
        else:
            path = Path(
                structure_path
                or Path(__file__).resolve().parents[2]
                / "output"
                / "progress_structure.json"
            )
            with path.open("r", encoding="utf-8") as file:
                self.structure = json.load(file)
        self._add_authored_curriculum_items()
        self._add_chatbot_practice_items()

    def _add_authored_curriculum_items(self):
        """Repair older progress blueprints from the authored content export."""
        path = Path(__file__).resolve().parents[2] / "output" / "content_skills.json"
        if not path.exists():
            return
        try:
            with path.open("r", encoding="utf-8") as file:
                skills = json.load(file)
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(skills, list):
            return

        chapters = {
            chapter.get("chapter_id"): chapter
            for chapter in self.structure.get("chapters", [])
            if chapter.get("chapter_id")
        }
        for skill in skills:
            skill_id = skill.get("skill_id")
            if skill_id not in CANONICAL_OBJECTIVE_IDS:
                continue
            chapter = chapters.get(skill_id)
            if not chapter:
                continue
            objectives = {
                objective.get("objective_id"): objective
                for objective in chapter.get("objectives", [])
                if objective.get("objective_id")
            }
            for authored in skill.get("objectives", []) or []:
                objective_id = authored.get("objective_id")
                if objective_id not in set(
                    CANONICAL_OBJECTIVE_IDS[skill_id].values()
                ):
                    continue
                objective = objectives.get(objective_id)
                if not objective:
                    continue
                existing_ids = {
                    item.get("item_id")
                    for item in objective.setdefault("items", [])
                }
                sources = [
                    ("text", authored.get("learning_content", [])),
                    ("activity", authored.get("activities", [])),
                    ("video", authored.get("videos", [])),
                ]
                if isinstance(authored.get("quiz"), dict):
                    sources.append(("quiz", [authored["quiz"]]))
                if isinstance(authored.get("reflection"), dict):
                    sources.append(("reflection", [authored["reflection"]]))
                for item_type, entries in sources:
                    for entry in entries or []:
                        if not isinstance(entry, dict) or not entry.get("item_id"):
                            continue
                        if entry["item_id"] in existing_ids:
                            existing = next(
                                item
                                for item in objective["items"]
                                if item.get("item_id") == entry["item_id"]
                            )
                            authored_content = (
                                entry.get("content")
                                or entry.get("instructions")
                                or entry.get("description")
                                or entry.get("question")
                                or " ".join(entry.get("steps") or [])
                            )
                            if len(authored_content) > len(
                                str(existing.get("content") or "")
                            ):
                                existing["content"] = authored_content
                            if entry.get("title") and (
                                not existing.get("title")
                                or len(entry["title"]) > len(
                                    str(existing.get("title") or "")
                                )
                            ):
                                existing["title"] = entry["title"]
                            continue
                        objective["items"].append({
                            **entry,
                            "item_id": entry["item_id"],
                            "type": item_type,
                            "title": entry.get("title") or entry["item_id"],
                            "content": (
                                entry.get("content")
                                or entry.get("description")
                                or entry.get("question")
                                or ""
                            ),
                            "objective_id": objective["objective_id"],
                            "skill_id": chapter["chapter_id"],
                        })
                        existing_ids.add(entry["item_id"])
                for entry in authored.get("activities", []) or []:
                    if not isinstance(entry, dict):
                        continue
                    existing = next(
                        (
                            item
                            for item in objective["items"]
                            if item.get("item_id") == entry.get("item_id")
                        ),
                        None,
                    )
                    if existing is None:
                        continue
                    authored_content = (
                        entry.get("content")
                        or entry.get("instructions")
                        or entry.get("description")
                        or entry.get("question")
                        or " ".join(entry.get("steps") or [])
                    )
                    if authored_content:
                        existing["content"] = authored_content
                    if entry.get("title"):
                        existing["title"] = entry["title"]
                required = [
                    item_id
                    for item_id in authored.get("required_items", [])
                    if item_id in existing_ids
                ]
                if required:
                    objective["required_item_ids"] = required

    def _add_chatbot_practice_items(self):
        """Register chatbot-created practices as trackable objective items."""
        for chapter in self.structure.get("chapters", []):
            for objective in chapter.get("objectives", []):
                practice = CHATBOT_PRACTICES.get(objective.get("objective_id"))
                if not practice:
                    continue
                items = objective.setdefault("items", [])
                if any(item.get("item_id") == practice["item_id"] for item in items):
                    continue
                metadata = activity_metadata(practice)
                items.append({
                    **practice,
                    "item_id": practice["item_id"],
                    "skill_id": chapter.get("chapter_id"),
                    "objective_id": objective.get("objective_id"),
                    **metadata,
                })
                required_ids = objective.setdefault("required_item_ids", [])
                if practice["item_id"] not in required_ids:
                    required_ids.append(practice["item_id"])

    def get_structure(self):
        return self.structure

    @staticmethod
    def _percentage(completed, total):
        if not total:
            return 0.0
        return round(completed / total * 100, 1)

    @staticmethod
    def _average(values):
        if not values:
            return 0.0
        return round(sum(values) / len(values), 1)

    @staticmethod
    def _progress_item_ids(objective):
        item_ids = {
            item["item_id"]
            for item in objective.get("items", [])
            if item.get("item_id")
        }
        required_item_ids = set(objective.get("required_item_ids", []))
        return required_item_ids & item_ids if required_item_ids else item_ids

    @staticmethod
    def completion_requirements(objective):
        """Return activity conditions and the single canonical quiz threshold."""
        activity_item_ids = []
        quiz_passing_scores = {}
        for item in objective.get("items", []):
            item_id = item.get("item_id")
            if not item_id:
                continue
            item_type = str(item.get("type") or "").lower()
            if item_type in {"activity", "practice", "reflection"}:
                activity_item_ids.append(item_id)
            elif item_type in {"quiz", "assessment"}:
                quiz_passing_scores[item_id] = (
                    ProgressService.DEFAULT_QUIZ_PASSING_SCORE
                )
        return {
            "activity_item_ids": activity_item_ids,
            "quiz_passing_scores": quiz_passing_scores,
        }

    @classmethod
    def completion_state(cls, objective, completed_item_ids, quiz_scores):
        """Evaluate activity and quiz completion from saved item results."""
        requirements = cls.completion_requirements(objective)
        completed_item_ids = set(completed_item_ids or [])
        quiz_scores = quiz_scores or {}
        activity_completed = any(
            item_id in completed_item_ids
            for item_id in requirements["activity_item_ids"]
        )
        passed_quiz_ids = {
            item_id
            for item_id, passing_score in requirements["quiz_passing_scores"].items()
            if item_id in completed_item_ids
            and isinstance(quiz_scores.get(item_id), (int, float))
            and not isinstance(quiz_scores.get(item_id), bool)
            and quiz_scores[item_id] >= passing_score
        }
        quiz_passed = bool(passed_quiz_ids)
        progress_item_ids = {
            item.get("item_id")
            for item in objective.get("items", [])
            if item.get("item_id")
        }
        coverage_percent = cls._percentage(
            len(progress_item_ids & completed_item_ids),
            len(progress_item_ids),
        )
        activity_eligible = (
            coverage_percent >= cls.ACTIVITY_ELIGIBILITY_PERCENT
            and not activity_completed
        )
        missing_item_ids = []
        if not activity_completed:
            missing_item_ids.extend(requirements["activity_item_ids"])
        if not quiz_passed:
            missing_item_ids.extend(requirements["quiz_passing_scores"])
        return {
            **requirements,
            "activity_completed": activity_completed,
            "quiz_passed": quiz_passed,
            "coverage_percent": coverage_percent,
            "activity_eligible": activity_eligible,
            "completed": activity_completed and quiz_passed,
            "required_item_ids": sorted(
                set(requirements["activity_item_ids"])
                | set(requirements["quiz_passing_scores"])
            ),
            "missing_required_item_ids": sorted(set(missing_item_ids)),
        }

    def calculate(self, progress_documents):
        """Return objective-based progress for the curriculum."""

        progress_documents = list(progress_documents)
        skills = [] 
        skill_percentages = []

        for chapter in self.structure.get("chapters", []):
            skill_id = chapter["chapter_id"]
            progress = next(
                (
                    document
                    for document in progress_documents
                    if document.get("skill_id") == skill_id
                ),
                {},
            )
            completed_ids = set(progress.get("completed_item_ids", []))
            completed_objective_ids = set(
                progress.get("completed_objective_ids", [])
            )
            quiz_scores = progress.get("quiz_scores", {})
            objectives = []
            objective_percentages = []

            for objective in chapter.get("objectives", []):
                progress_item_ids = self._progress_item_ids(objective)
                state = self.completion_state(
                    objective,
                    completed_ids,
                    quiz_scores,
                )
                is_completed = (
                    state["completed"]
                    or objective["objective_id"] in completed_objective_ids
                )
                if objective["objective_id"] in completed_objective_ids:
                    state = {
                        **state,
                        "completed": True,
                        "missing_required_item_ids": [],
                    }
                completed_count = len(progress_item_ids & completed_ids)
                total_count = len(progress_item_ids)
                if is_completed:
                    objective_progress = 100.0
                else:
                    objective_progress = self._percentage(
                        completed_count,
                        total_count,
                    )

                objective_percentages.append(objective_progress)

                objectives.append({
                    "objective_id": objective["objective_id"],
                    "title": objective["title"],
                    "progress": objective_progress,
                    "completed_items": completed_count,
                    "total_items": total_count,
                    "completed": is_completed,
                    "activity_completed": state["activity_completed"],
                    "quiz_passed": state["quiz_passed"],
                    "required_item_ids": state["required_item_ids"],
                    "missing_required_item_ids": state[
                        "missing_required_item_ids"
                    ],
                })

            skill_progress = self._average(objective_percentages)
            skill_percentages.append(skill_progress)
            skills.append({
                "skill_id": skill_id,
                "title": chapter.get("title", ""),
                "progress": skill_progress,
                "completed_items": sum(
                    objective_progress == 100.0
                    for objective_progress in objective_percentages
                ),
                "total_items": len(objective_percentages),
                "objectives": objectives,
            })

        return {
            "overall_progress": self._average(skill_percentages),
            "completed_items": sum(
                skill_progress == 100.0
                for skill_progress in skill_percentages
            ),
            "total_learning_items": len(skill_percentages),
            "skills": skills,
        }

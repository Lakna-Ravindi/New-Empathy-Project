"""Calculate learning progress from the assigned curriculum structure."""

import json
from pathlib import Path

from learning.chatbot_practice import CHATBOT_PRACTICES


class ProgressService:
    """Build objective, skill, and overall progress summaries."""

    DEFAULT_QUIZ_PASSING_SCORE = 80

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
        self._add_chatbot_practice_items()

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
                items.append({
                    **practice,
                    "item_id": practice["item_id"],
                    "skill_id": chapter.get("chapter_id"),
                    "objective_id": objective.get("objective_id"),
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
        """Return the authored activity and quiz conditions for an objective."""
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
                passing_score = item.get("passing_score")
                if (
                    isinstance(passing_score, bool)
                    or not isinstance(passing_score, (int, float))
                    or not 0 <= passing_score <= 100
                ):
                    passing_score = ProgressService.DEFAULT_QUIZ_PASSING_SCORE
                quiz_passing_scores[item_id] = passing_score
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
        missing_item_ids = []
        if not activity_completed:
            missing_item_ids.extend(requirements["activity_item_ids"])
        if not quiz_passed:
            missing_item_ids.extend(requirements["quiz_passing_scores"])
        return {
            **requirements,
            "activity_completed": activity_completed,
            "quiz_passed": quiz_passed,
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

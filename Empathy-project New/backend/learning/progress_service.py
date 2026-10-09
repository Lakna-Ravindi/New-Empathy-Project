"""Calculate learning progress from the assigned curriculum structure."""

import json
from pathlib import Path


class ProgressService:
    """Build objective, skill, and overall progress summaries."""

    def __init__(self, structure_path=None, structure=None):
        if structure is not None:
            self.structure = structure
            return

        path = Path(
            structure_path
            or Path(__file__).resolve().parents[2]
            / "output"
            / "progress_structure.json"
        )
        with path.open("r", encoding="utf-8") as file:
            self.structure = json.load(file)

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
    def _required_item_ids(objective):
        item_ids = {
            item["item_id"]
            for item in objective.get("items", [])
            if item.get("item_id")
        }
        required_item_ids = set(objective.get("required_item_ids", []))
        return required_item_ids & item_ids if required_item_ids else item_ids

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
            objectives = []
            objective_percentages = []

            for objective in chapter.get("objectives", []):
                item_ids = self._required_item_ids(objective)
                completed_count = len(item_ids & completed_ids)
                total_count = len(item_ids)

                if objective["objective_id"] in completed_objective_ids:
                    objective_progress = 100.0
                elif total_count:
                    objective_progress = self._percentage(
                        completed_count,
                        total_count,
                    )
                else:
                    objective_progress = 100.0 if (
                        objective["objective_id"] in completed_objective_ids
                    ) else 0.0

                objective_percentages.append(objective_progress)

                objectives.append({
                    "objective_id": objective["objective_id"],
                    "title": objective["title"],
                    "progress": objective_progress,
                    "completed_items": completed_count,
                    "total_items": total_count,
                    "completed": objective_progress == 100.0,
                })

            skill_progress = self._average(objective_percentages)
            skill_percentages.append(skill_progress)
            skills.append({
                "skill_id": skill_id,
                "title": chapter["title"],
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

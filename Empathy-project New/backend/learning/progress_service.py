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

    def calculate(self, progress_documents):
        """Return progress for all assigned items in the curriculum."""

        progress_documents = list(progress_documents)
        progress_by_skill = {
            document.get("skill_id"): set(document.get("completed_item_ids", []))
            for document in progress_documents
        }
        skills = []
        overall_total = 0
        overall_completed = 0

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
            skill_total = 0
            skill_completed = 0

            for objective in chapter.get("objectives", []):
                item_ids = {
                    item["item_id"]
                    for item in objective.get("items", [])
                }
                completed_count = len(item_ids & completed_ids)
                total_count = len(item_ids)

                if total_count:
                    objective_progress = self._percentage(
                        completed_count,
                        total_count,
                    )
                    skill_total += total_count
                    skill_completed += completed_count
                else:
                    objective_progress = 100.0 if (
                        objective["objective_id"] in completed_objective_ids
                    ) else 0.0
                    skill_total += 1
                    skill_completed += int(objective_progress == 100.0)

                objectives.append({
                    "objective_id": objective["objective_id"],
                    "title": objective["title"],
                    "progress": objective_progress,
                    "completed_items": completed_count,
                    "total_items": total_count,
                    "completed": objective_progress == 100.0,
                })

            overall_total += skill_total
            overall_completed += skill_completed
            skills.append({
                "skill_id": skill_id,
                "title": chapter["title"],
                "progress": self._percentage(skill_completed, skill_total),
                "completed_items": skill_completed,
                "total_items": skill_total,
                "objectives": objectives,
            })

        return {
            "overall_progress": self._percentage(
                overall_completed,
                overall_total,
            ),
            "completed_items": overall_completed,
            "total_learning_items": overall_total,
            "skills": skills,
        }

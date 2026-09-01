"""MongoDB persistence for learning interactions, progress, and review data."""

from datetime import datetime, timezone
import os
import uuid

from bson import ObjectId
from pymongo import MongoClient


class LearningStore:
    """Stores application data; it never makes learning decisions."""

    def __init__(self, uri=None, database_name=None):
        self.is_available = True
        self.local_storage = {
            "interactions": [],
            "progress": {},
            "evaluations": []
        }
        
        try:
            self.client = MongoClient(
                uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
                serverSelectionTimeoutMS=5000,
            )
            self.db = self.client[
                database_name or os.getenv("MONGODB_DATABASE", "empathy_learning")
            ]
            self.interactions = self.db["learning_interactions"]
            self.progress = self.db["student_progress"]
            self.evaluations = self.db["controller_evaluations"]
        except Exception as e:
            self.is_available = False
            print(f"WARNING: MongoDB unavailable, using local fallback storage: {e}")

    @staticmethod
    def _now():
        return datetime.now(timezone.utc)

    def save_interaction(self, student_id, learning_context, educational_response):
        """Store the controller decision and generated wording as one audit record."""
        document = {
            "student_id": student_id,
            "question": learning_context["student_question"],
            "detected_emotion": learning_context.get("detected_emotion", "unknown"),
            "detected_terms": learning_context.get("detected_terms", []),
            "status": learning_context["status"],
            "selected_skill": learning_context.get("skill"),
            "learning_objective": learning_context.get("learning_objective"),
            "recommended_activity": learning_context.get("recommended_activity"),
            "response": educational_response,
            "timestamp": self._now(),
        }
        
        if not self.is_available:
            # Fallback: store locally with a local- prefix for ID
            local_id = f"local-{uuid.uuid4()}"
            document["_id"] = local_id
            self.local_storage["interactions"].append(document)
            return local_id
        
        result = self.interactions.insert_one(document)
        return str(result.inserted_id)

    def get_progress(self, student_id, skill_id):
        key = f"{student_id}:{skill_id}"
        
        if not self.is_available:
            # Fallback: get from local storage
            return self.local_storage["progress"].get(key, {
                "student_id": student_id,
                "skill_id": skill_id,
                "completed_objective_ids": [],
                "next_recommended_learning_objective": None,
            })
        
        document = self.progress.find_one(
            {"student_id": student_id, "skill_id": skill_id},
            {"_id": 0},
        )
        return document or {
            "student_id": student_id,
            "skill_id": skill_id,
            "completed_objective_ids": [],
            "next_recommended_learning_objective": None,
        }

    def complete_objective(self, student_id, skill_id, objective, next_objective):
        """Mark only an explicitly confirmed objective as complete."""
        key = f"{student_id}:{skill_id}"
        now = self._now()
        
        if not self.is_available:
            # Fallback: update local storage
            if key not in self.local_storage["progress"]:
                self.local_storage["progress"][key] = {
                    "student_id": student_id,
                    "skill_id": skill_id,
                    "completed_objective_ids": [],
                    "next_recommended_learning_objective": None,
                    "created_at": now,
                }
            self.local_storage["progress"][key]["completed_objective_ids"].append(objective["id"])
            self.local_storage["progress"][key]["next_recommended_learning_objective"] = next_objective
            self.local_storage["progress"][key]["updated_at"] = now
            return self.local_storage["progress"][key]
        
        self.progress.update_one(
            {"student_id": student_id, "skill_id": skill_id},
            {
                "$set": {
                    "next_recommended_learning_objective": next_objective,
                    "updated_at": now,
                },
                "$addToSet": {"completed_objective_ids": objective["id"]},
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )
        return self.get_progress(student_id, skill_id)

    def save_evaluation(self, interaction_id, reviewer_id, evaluation):
        """Save a human review; automated self-judging would not be reliable."""
        document = {
            "interaction_id": ObjectId(interaction_id) if not interaction_id.startswith("local-") else interaction_id,
            "reviewer_id": reviewer_id,
            "correct_skill_selected": evaluation["correct_skill_selected"],
            "activity_matches_emotion": evaluation["activity_matches_emotion"],
            "response_educationally_aligned": evaluation["response_educationally_aligned"],
            "notes": evaluation.get("notes", ""),
            "timestamp": self._now(),
        }
        
        if not self.is_available:
            # Fallback: store locally
            local_id = f"local-{uuid.uuid4()}"
            document["_id"] = local_id
            self.local_storage["evaluations"].append(document)
            return local_id
        
        result = self.evaluations.insert_one(document)
        return str(result.inserted_id)

    def evaluation_summary(self):
        """Return human-review percentages for controller quality monitoring."""
        
        if not self.is_available:
            # Fallback: calculate from local storage
            evaluations = self.local_storage["evaluations"]
            if not evaluations:
                return {"reviewed_interactions": 0}
            
            count = len(evaluations)
            correct_count = sum(1 for e in evaluations if e.get("correct_skill_selected"))
            activity_match_count = sum(1 for e in evaluations if e.get("activity_matches_emotion"))
            alignment_count = sum(1 for e in evaluations if e.get("response_educationally_aligned"))
            
            return {
                "reviewed_interactions": count,
                "correct_skill_selection_rate": round((correct_count / count * 100), 1) if count else 0,
                "activity_emotion_match_rate": round((activity_match_count / count * 100), 1) if count else 0,
                "educational_alignment_rate": round((alignment_count / count * 100), 1) if count else 0,
            }
        
        pipeline = [{
            "$group": {
                "_id": None,
                "reviewed_interactions": {"$sum": 1},
                "correct_skill_selection_rate": {
                    "$avg": {"$cond": ["$correct_skill_selected", 1, 0]}
                },
                "activity_emotion_match_rate": {
                    "$avg": {"$cond": ["$activity_matches_emotion", 1, 0]}
                },
                "educational_alignment_rate": {
                    "$avg": {"$cond": ["$response_educationally_aligned", 1, 0]}
                },
            }
        }]
        summary = next(self.evaluations.aggregate(pipeline), None)
        if not summary:
            return {"reviewed_interactions": 0}

        summary.pop("_id", None)
        for key in (
            "correct_skill_selection_rate",
            "activity_emotion_match_rate",
            "educational_alignment_rate",
        ):
            summary[key] = round(summary[key] * 100, 1)
        return summary



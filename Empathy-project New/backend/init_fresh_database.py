"""Initialize the fresh Empathy MongoDB database without destructive commands.

This script only creates application collections and indexes. It never drops,
deletes, replaces, or updates documents.
"""

from __future__ import annotations

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import CollectionInvalid, PyMongoError

from mongo_config import MONGODB_DATABASE, MONGODB_URI


COLLECTIONS = (
    "users",
    "screening_results",
    "pre_skill_assessments",
    "learning_interactions",
    "student_progress",
    "controller_evaluations",
    "safety_events",
    "student_safety_status",
)


def ensure_collection(db, name: str) -> None:
    if name in db.list_collection_names():
        return
    try:
        db.create_collection(name)
    except CollectionInvalid:
        # Another process may have created it between the listing and creation.
        pass


def initialize_database() -> dict[str, object]:
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
        db = client[MONGODB_DATABASE]
        for collection in COLLECTIONS:
            ensure_collection(db, collection)

        indexes = {
            "users": [
                ([("email", ASCENDING)], {"unique": True, "name": "uq_users_email"}),
                ([("username", ASCENDING)], {"unique": True, "name": "uq_users_username"}),
                ([("student_id", ASCENDING)], {
                    "unique": True,
                    "sparse": True,
                    "name": "uq_users_student_id",
                }),
            ],
            "screening_results": [
                ([("student_id", ASCENDING)], {
                    "unique": True,
                    "name": "uq_screening_student_id",
                }),
            ],
            "pre_skill_assessments": [
                ([("student_id", ASCENDING)], {
                    "unique": True,
                    "name": "uq_pre_skill_student_id",
                }),
            ],
            "learning_interactions": [
                ([("student_id", ASCENDING), ("timestamp", DESCENDING)], {
                    "name": "ix_interactions_student_timestamp",
                }),
            ],
            "student_progress": [
                ([("student_id", ASCENDING), ("skill_id", ASCENDING)], {
                    "unique": True,
                    "name": "uq_progress_student_skill",
                }),
            ],
            "controller_evaluations": [
                ([("interaction_id", ASCENDING)], {
                    "name": "ix_evaluations_interaction_id",
                }),
                ([("reviewer_id", ASCENDING), ("timestamp", DESCENDING)], {
                    "name": "ix_evaluations_reviewer_timestamp",
                }),
            ],
            "safety_events": [
                ([("student_id", ASCENDING), ("request_id", ASCENDING)], {
                    "unique": True,
                    "name": "uq_safety_event_request",
                }),
                ([("student_id", ASCENDING), ("created_at", DESCENDING)], {
                    "name": "ix_safety_event_student_created",
                }),
            ],
            "student_safety_status": [
                ([("student_id", ASCENDING)], {
                    "unique": True,
                    "name": "uq_student_safety_status",
                }),
            ],
        }

        created_indexes = []
        for collection_name, definitions in indexes.items():
            collection = db[collection_name]
            for keys, options in definitions:
                collection.create_index(keys, **options)
                created_indexes.append(f"{collection_name}.{options['name']}")

        return {
            "database": MONGODB_DATABASE,
            "collections": list(COLLECTIONS),
            "indexes": created_indexes,
        }
    except PyMongoError:
        raise
    finally:
        client.close()


if __name__ == "__main__":
    result = initialize_database()
    print(f"Initialized MongoDB database '{result['database']}'.")
    print(f"Collections ensured: {', '.join(result['collections'])}")
    print(f"Indexes ensured: {len(result['indexes'])}")

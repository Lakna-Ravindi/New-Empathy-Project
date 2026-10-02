"""Import a JSON object or list into MongoDB using the backend environment config."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient


BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(Path(__file__).resolve().with_name(".env"))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def documents_from(data: Any, document_id: str) -> list[dict[str, Any]]:
    if isinstance(data, list):
        documents = []
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"List item {index} must be a JSON object")
            documents.append(item)
        return documents

    if isinstance(data, dict):
        return [{"_id": document_id, **data}]

    raise ValueError("JSON root must be an object or a list of objects")


def save_json(
    input_path: Path,
    collection_name: str,
    document_id: str,
    key_field: str,
) -> int:
    data = load_json(input_path)
    documents = documents_from(data, document_id)
    if not documents:
        print("No documents to import.")
        return 0

    client = MongoClient(
        os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
        serverSelectionTimeoutMS=5000,
    )
    client.admin.command("ping")

    database_name = os.getenv("MONGODB_DATABASE", "empathy_learning")
    collection = client[database_name][collection_name]
    imported = 0

    try:
        for index, document in enumerate(documents):
            value = document.get(key_field)
            if value is None:
                if len(documents) == 1 and document.get("_id"):
                    value = document["_id"]
                else:
                    raise ValueError(
                        f"Document {index} has no '{key_field}' field"
                    )

            stored = {**document, "_id": str(value)}
            collection.replace_one({"_id": stored["_id"]}, stored, upsert=True)
            imported += 1
    finally:
        client.close()

    return imported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--document-id", default="json_document")
    parser.add_argument("--key-field", default="id")
    args = parser.parse_args()

    imported = save_json(
        args.input,
        args.collection,
        args.document_id,
        args.key_field,
    )
    print(f"Imported {imported} document(s) into '{args.collection}'.")


if __name__ == "__main__":
    main()
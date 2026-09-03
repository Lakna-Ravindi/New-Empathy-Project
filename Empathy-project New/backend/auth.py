import os
from datetime import datetime, timedelta, timezone
from functools import wraps

import bcrypt
import jwt
from bson import ObjectId
from dotenv import load_dotenv
from flask import g, jsonify, request
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError


load_dotenv()

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "24"))
JWT_ALGORITHM = "HS256"


class StudentStore:
    def __init__(self):
        mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        database_name = os.getenv("MONGODB_DATABASE", "empathy_learning")

        self.client = MongoClient(
            mongo_uri,
            serverSelectionTimeoutMS=5000,
        )
        self.client.admin.command("ping")

        self.db = self.client[database_name]
        self.students = self.db["students"]

        # Same email cannot be registered twice.
        self.students.create_index("email", unique=True)

    def create_student(self, name, email, password):
        password_hash = bcrypt.hashpw(
            password.encode("utf-8"),
            bcrypt.gensalt(),
        ).decode("utf-8")

        student = {
            "name": name,
            "email": email.lower(),
            "password_hash": password_hash,
            "created_at": datetime.now(timezone.utc),
        }

        result = self.students.insert_one(student)

        return {
            "id": str(result.inserted_id),
            "name": name,
            "email": email.lower(),
        }

    def find_by_email(self, email):
        return self.students.find_one({"email": email.lower()})

    def find_by_id(self, student_id):
        try:
            return self.students.find_one({"_id": ObjectId(student_id)})
        except Exception:
            return None


def create_access_token(student):
    if not JWT_SECRET_KEY:
        raise RuntimeError("JWT_SECRET_KEY is missing from backend/.env")

    now = datetime.now(timezone.utc)

    payload = {
        "sub": str(student["_id"]),
        "email": student["email"],
        "iat": now,
        "exp": now + timedelta(hours=JWT_EXPIRY_HOURS),
    }

    return jwt.encode(
        payload,
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )


def require_auth(view_function):
    @wraps(view_function)
    def wrapped(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")

        if not auth_header.startswith("Bearer "):
            return jsonify({
                "error": "Authentication is required."
            }), 401

        token = auth_header.split(" ", 1)[1].strip()

        try:
            payload = jwt.decode(
                token,
                JWT_SECRET_KEY,
                algorithms=[JWT_ALGORITHM],
            )

            # Logged-in student's ID is available to protected routes.
            g.student_id = payload["sub"]

        except jwt.ExpiredSignatureError:
            return jsonify({
                "error": "Your session has expired. Please log in again."
            }), 401

        except (jwt.InvalidTokenError, KeyError):
            return jsonify({
                "error": "Invalid authentication token."
            }), 401

        return view_function(*args, **kwargs)

    return wrapped
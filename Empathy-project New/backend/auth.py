import os
from datetime import datetime, timedelta, timezone
from functools import wraps

import bcrypt
import jwt
from bson import ObjectId
from bson.errors import InvalidId
from dotenv import load_dotenv
from flask import g, jsonify, request
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError


load_dotenv()

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "24"))
JWT_ALGORITHM = "HS256"


VALID_GENDERS = {"Male", "Female", "Prefer not to say"}
VALID_AGE_GROUPS = {
    "Below 16 years",
    "16-18 years",
    "19-21 years",
    "22-25 years",
    "26+ years",
}
VALID_ROLES = {"student", "admin"}


class UserStore:
    def __init__(self):
        self.is_available = False
        self.client = None
        self.db = None
        self.users = None
        self.mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self.database_name = os.getenv("MONGODB_DATABASE", "empathy_learning")

        self._connect()

    def _connect(self):
        self.client = MongoClient(
            self.mongo_uri,
            serverSelectionTimeoutMS=5000,
        )

        try:
            self.client.admin.command("ping")
            self.db = self.client[self.database_name]
            self.users = self.db["users"]

            # Usernames and emails must both be unique.
            self.users.create_index("email", unique=True)
            self.users.create_index("username", unique=True)
            self.is_available = True
        except PyMongoError as error:
            self.is_available = False
            print(f"WARNING: MongoDB unavailable for authentication: {error}")

    def _require_database(self):
        if not self.is_available:
            self._connect()
        if not self.is_available or self.users is None:
            raise PyMongoError("MongoDB authentication database is unavailable.")

    def create_user(self, name, username, email, password, role="student", gender=None, age_group=None):
        self._require_database()
        if role not in VALID_ROLES:
            raise ValueError("Invalid role")
        password_hash = bcrypt.hashpw(
            password.encode("utf-8"),
            bcrypt.gensalt(),
        ).decode("utf-8")

        user = {
            "name": name,
            "username": username.lower(),
            "email": email.lower(),
            "password_hash": password_hash,
            "role": role,
            "created_at": datetime.now(timezone.utc),
        }
        if role == "student":
            user["gender"] = gender
            user["age_group"] = age_group

        result = self.users.insert_one(user)

        return {
            "id": str(result.inserted_id),
            "name": name,
            "username": username.lower(),
            "email": email.lower(),
            "role": role,
            **({"gender": gender, "age_group": age_group} if role == "student" else {}),
        }

    def create_student(self, name, username, email, password, gender, age_group):
        return self.create_user(name, username, email, password, "student", gender, age_group)

    def create_admin(self, name, username, email, password):
        return self.create_user(name, username, email, password, "admin")

    def find_by_username(self, username):
        self._require_database()
        return self.users.find_one({"username": username.lower()})

    def find_by_email(self, email):
        self._require_database()
        return self.users.find_one({"email": email.lower()})

    def find_by_id(self, student_id):
        self._require_database()
        return self.users.find_one({"_id": ObjectId(student_id)})

    def list_users(self):
        self._require_database()
        return list(self.users.find({}).sort("created_at", 1))

    def update_user(self, user_id, changes):
        self._require_database()
        result = self.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": changes},
        )
        return result.modified_count > 0 or result.matched_count > 0

    def delete_user(self, user_id):
        self._require_database()
        result = self.users.delete_one({"_id": ObjectId(user_id)})
        return result.deleted_count > 0


StudentStore = UserStore


_user_store = None


def get_user_store():
    global _user_store
    if _user_store is None:
        _user_store = UserStore()
    return _user_store


def create_access_token(student):
    if not JWT_SECRET_KEY:
        raise RuntimeError("JWT_SECRET_KEY is missing from backend/.env")

    now = datetime.now(timezone.utc)

    payload = {
        "sub": str(student["_id"]),
        "username": student["username"],
        "email": student["email"],
        "role": student.get("role", "student"),
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

            # The authenticated user's identity is available to protected routes.
            g.student_id = payload["sub"]
            user = get_user_store().find_by_id(g.student_id)
            if not user:
                return jsonify({"error": "User account was not found."}), 401
            g.current_user = user
            g.user_role = user.get("role")

        except jwt.ExpiredSignatureError:
            return jsonify({
                "error": "Your session has expired. Please log in again."
            }), 401

        except (jwt.InvalidTokenError, KeyError, InvalidId, PyMongoError):
            return jsonify({
                "error": "Invalid authentication token."
            }), 401

        return view_function(*args, **kwargs)

    return wrapped


def require_role(*allowed_roles):
    def decorator(view_function):
        @wraps(view_function)
        @require_auth
        def wrapped(*args, **kwargs):
            if g.user_role not in allowed_roles:
                return jsonify({"error": "Admin access required" if allowed_roles == ("admin",) else "You do not have permission to perform this action."}), 403
            return view_function(*args, **kwargs)

        return wrapped

    return decorator
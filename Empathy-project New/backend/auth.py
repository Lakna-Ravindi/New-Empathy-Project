import os
import logging
import hashlib
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path

import secrets
import bcrypt
import jwt
from bson import ObjectId
from bson.errors import InvalidId
from flask import g, jsonify, request
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError
from dotenv import load_dotenv
from mongo_config import MONGODB_DATABASE, MONGODB_URI

# Authentication can be imported without server.py, so load the backend
# configuration before reading JWT settings.
load_dotenv(Path(__file__).resolve().parent / ".env")

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "24"))
JWT_ALGORITHM = "HS256"
logger = logging.getLogger(__name__)

INSECURE_JWT_SECRETS = {
    "fallback-secret-key-change-in-production",
    "secret",
    "changeme",
    "your-secret-key",
    "default",
    "test",
    "admin",
    "password",
    "123456",
    "jwt_secret",
}


def is_production_mode() -> bool:
    env = (os.getenv("FLASK_ENV") or os.getenv("APP_ENV") or os.getenv("ENVIRONMENT") or "").lower()
    return env in {"production", "prod"}


def validate_jwt_secret(secret: str | None = None, is_production: bool | None = None) -> str:
    """Validate that JWT_SECRET_KEY is configured securely, particularly in production."""
    if is_production is None:
        is_production = is_production_mode()

    if secret is None:
        secret = os.getenv("JWT_SECRET_KEY")

    if is_production:
        if not secret or not secret.strip():
            raise RuntimeError("CRITICAL: JWT_SECRET_KEY is not configured in production environment.")
        clean = secret.strip()
        if clean.lower() in INSECURE_JWT_SECRETS:
            raise RuntimeError(f"CRITICAL: JWT_SECRET_KEY uses a known insecure default '{clean}'.")
        if len(clean) < 32:
            raise RuntimeError(f"CRITICAL: JWT_SECRET_KEY must be at least 32 characters in production (got {len(clean)}).")
        return clean

    # Development / testing fallback
    if not secret or not secret.strip():
        logger.warning("JWT_SECRET_KEY is not configured. Using development secret.")
        return "dev-insecure-secret-key-do-not-use-in-production-12345"

    return secret.strip()


def get_jwt_secret() -> str:
    """Get the active JWT secret, validating against environment configuration."""
    global JWT_SECRET_KEY
    val = JWT_SECRET_KEY or os.getenv("JWT_SECRET_KEY")
    if is_production_mode():
        return validate_jwt_secret(val, is_production=True)
    if val:
        return val
    return validate_jwt_secret(val, is_production=False)


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
        self.mongo_uri = MONGODB_URI
        self.database_name = MONGODB_DATABASE

        self._connect()

    def _connect(self):
        try:
            self.client = MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=5000,
            )
            self.client.admin.command("ping")
            self.db = self.client[self.database_name]
            self.users = self.db["users"]

            indexes = (
                ([("email", 1)], {"unique": True, "name": "uq_users_email"}),
                ([("username", 1)], {"unique": True, "name": "uq_users_username"}),
                (
                    [("student_id", 1)],
                    {
                        "unique": True,
                        "sparse": True,
                        "name": "uq_users_student_id",
                    },
                ),
            )
            existing_indexes = self.users.index_information()
            for keys, options in indexes:
                compatible = any(
                    tuple(index.get("key", [])) == tuple(keys)
                    and index.get("unique", False) == options["unique"]
                    and index.get("sparse", False) == options.get("sparse", False)
                    for index in existing_indexes.values()
                )
                if not compatible:
                    self.users.create_index(keys, **options)

            if self.db is not None:
                revoked_coll = self.db["revoked_tokens"]
                rev_indexes = revoked_coll.index_information()
                if not any(tuple(idx.get("key", [])) == (("expires_at", 1),) for idx in rev_indexes.values()):
                    revoked_coll.create_index([("expires_at", 1)], expireAfterSeconds=0, name="idx_revoked_tokens_ttl")
                if not any(tuple(idx.get("key", [])) == (("token_hash", 1),) for idx in rev_indexes.values()):
                    revoked_coll.create_index([("token_hash", 1)], unique=True, name="uq_revoked_token_hash")

            self.is_available = True
        except PyMongoError as error:
            self.is_available = False
            logger.warning("MongoDB unavailable for authentication: %s", type(error).__name__)

    def revoke_token(self, token: str, exp_timestamp: float | int | None = None) -> bool:
        """Revoke a JWT token by recording its SHA256 hash."""
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        expires_at = (
            datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)
            if exp_timestamp
            else datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS)
        )
        if not self.is_available or self.db is None:
            if not hasattr(self, "_revoked_tokens_memory"):
                self._revoked_tokens_memory = set()
            self._revoked_tokens_memory.add(token_hash)
            return True

        try:
            revoked_coll = self.db["revoked_tokens"]
            revoked_coll.update_one(
                {"token_hash": token_hash},
                {
                    "$set": {
                        "token_hash": token_hash,
                        "revoked_at": datetime.now(timezone.utc),
                        "expires_at": expires_at,
                    }
                },
                upsert=True,
            )
            return True
        except PyMongoError as error:
            logger.warning("Failed to record revoked token in MongoDB: %s", error)
            if not hasattr(self, "_revoked_tokens_memory"):
                self._revoked_tokens_memory = set()
            self._revoked_tokens_memory.add(token_hash)
            return True

    def is_token_revoked(self, token: str) -> bool:
        """Check if a token hash exists in the revoked_tokens store."""
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        if hasattr(self, "_revoked_tokens_memory") and token_hash in self._revoked_tokens_memory:
            return True

        if not self.is_available or self.db is None:
            return False

        try:
            revoked_coll = self.db["revoked_tokens"]
            return revoked_coll.find_one({"token_hash": token_hash}) is not None
        except PyMongoError:
            return False

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
            "username": username.lower(),
            "email": email.lower(),
            "password_hash": password_hash,
            "role": role,
            "created_at": datetime.now(timezone.utc),
        }
        if name is not None:
            user["name"] = name
        if role == "student":
            user["student_id"] = f"STU-{secrets.token_hex(4).upper()}"
            user["gender"] = gender
            if age_group is not None:
                user["age_group"] = age_group

        result = self.users.insert_one(user)

        public_user = {
            "id": str(result.inserted_id),
            "username": username.lower(),
            "email": email.lower(),
            "role": role,
        }
        if role == "student":
            public_user["student_id"] = user["student_id"]
        return public_user

    def create_student(self, username, email, password, gender):
        return self.create_user(None, username, email, password, "student", gender)

    def create_admin(self, name, username, email, password):
        return self.create_user(name, username, email, password, "admin")

    def ensure_student_id(self, user):
        if user.get("role", "student") != "student" or user.get("student_id"):
            return user

        self.users.update_one(
            {"_id": user["_id"], "student_id": {"$exists": False}},
            {"$set": {"student_id": f"STU-{secrets.token_hex(4).upper()}"}},
        )
        return self.users.find_one({"_id": user["_id"]})

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
    secret = get_jwt_secret()
    if not secret:
        raise RuntimeError("JWT_SECRET_KEY is not configured in the environment.")

    now = datetime.now(timezone.utc)

    payload = {
        "sub": str(student["_id"]),
        "student_id": student.get("student_id", str(student["_id"])),
        "username": student["username"],
        "email": student["email"],
        "role": student.get("role", "student"),
        "iat": now,
        "exp": now + timedelta(hours=JWT_EXPIRY_HOURS),
    }

    return jwt.encode(
        payload,
        secret,
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

        secret = get_jwt_secret()
        if not secret:
            return jsonify({
                "error": "Authentication service is unavailable."
            }), 503

        # Check if token has been revoked
        user_store = get_user_store()
        if hasattr(user_store, "is_token_revoked") and user_store.is_token_revoked(token):
            return jsonify({
                "error": "Token has been revoked. Please log in again."
            }), 401

        try:
            payload = jwt.decode(
                token,
                secret,
                algorithms=[JWT_ALGORITHM],
            )

            # MongoDB _id authenticates the account; student_id identifies learning data.
            g.user_id = payload["sub"]
            user = get_user_store().find_by_id(g.user_id)
            if not user:
                return jsonify({"error": "User account was not found."}), 401
            user = get_user_store().ensure_student_id(user)
            g.current_user = user
            g.student_id = user.get("student_id") or payload.get("student_id") or g.user_id
            g.user_role = user.get("role")

        except jwt.ExpiredSignatureError:
            return jsonify({
                "error": "Your session has expired. Please log in again."
            }), 401

        except PyMongoError:
            return jsonify({
                "error": "Authentication service is unavailable."
            }), 503

        except (jwt.InvalidTokenError, KeyError, InvalidId):
            return jsonify({
                "error": "Invalid authentication token."
            }), 401

        return view_function(*args, **kwargs)

    return wrapped


def revoke_current_token() -> bool:
    """Revoke the JWT token from the Authorization header of the current request."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return False
    token = auth_header.split(" ", 1)[1].strip()
    try:
        secret = get_jwt_secret()
        payload = jwt.decode(token, secret, algorithms=[JWT_ALGORITHM], options={"verify_exp": False})
        exp = payload.get("exp")
        return get_user_store().revoke_token(token, exp)
    except Exception:
        return get_user_store().revoke_token(token)


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
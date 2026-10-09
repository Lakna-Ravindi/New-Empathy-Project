from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthCredentials
from app.core.security import decode_token, get_user_id_from_token
from learning.interaction_store import LearningStore
from bson import ObjectId

security = HTTPBearer()
learning_store = LearningStore()
users_collection = learning_store.db["users"] if learning_store.db else None


async def get_current_user(credentials: HTTPAuthCredentials = Depends(security)):
    """
    Dependency that validates JWT and returns current user.
    
    Usage in endpoint:
        @app.get("/api/me")
        async def get_me(current_user: dict = Depends(get_current_user)):
            return {"user": current_user}
    """
    token = credentials.credentials
    
    # Decode and verify token
    user_id = get_user_id_from_token(token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Fetch user from MongoDB
    if users_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="User database unavailable",
        )
    
    try:
        user = users_collection.find_one({"_id": ObjectId(user_id)})
    except:
        user = None
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    
    return user


async def get_current_student(current_user: dict = Depends(get_current_user)):
    """Dependency for student-only endpoints."""
    role = current_user.get("role", "student")
    if role != "student":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action is only available to students",
        )
    return current_user


async def get_current_reviewer(current_user: dict = Depends(get_current_user)):
    """Dependency for reviewer-only endpoints."""
    role = current_user.get("role", "student")
    if role not in ["reviewer", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions",
        )
    return current_user


async def get_current_admin(current_user: dict = Depends(get_current_user)):
    """Dependency for admin-only endpoints."""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user
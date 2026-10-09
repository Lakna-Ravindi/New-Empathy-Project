from datetime import datetime, timedelta, timezone
from typing import Optional
import jwt
from passlib.context import CryptContext
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
BASE_DIR = Path(__file__).resolve().parents[3]
load_dotenv(BASE_DIR / ".env")

# Password hashing configuration. Argon2 does not impose bcrypt's 72-byte limit.
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# JWT configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

# ==================== PASSWORD FUNCTIONS ====================

def hash_password(password: str) -> str:
    """Hash a password using Argon2."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

# ==================== JWT FUNCTIONS ====================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Dictionary to encode (should include 'sub' with user ID)
        expires_delta: Custom expiration time
    
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Optional[dict]:
    """
    Decode and verify a JWT token.
    
    Args:
        token: JWT token string
    
    Returns:
        Decoded payload if valid, None if invalid/expired
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None  # Token expired
    except jwt.InvalidTokenError:
        return None  # Invalid token


def get_user_id_from_token(token: str) -> Optional[str]:
    """Extract user ID from token's 'sub' claim."""
    payload = decode_token(token)
    if payload:
        return payload.get("sub")
    return None
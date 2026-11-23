from datetime import datetime, timedelta, timezone

from jose import jwt

from config.settings import settings


def create_access_token(sub: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(seconds=settings.session_expire_seconds)
    return jwt.encode({"sub": sub, "exp": expire}, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    # raises jose.JWTError on bad / expired token
    return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])

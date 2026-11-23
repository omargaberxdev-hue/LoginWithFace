from fastapi import Cookie, Depends, HTTPException, Request
from jose import JWTError
from sqlalchemy.orm import Session

from config import db as database
from config.users import User
from model.pipeline import FacePipeline
from utils.security import decode_access_token


def get_db():
    session = database.SessionLocal()
    try:
        yield session
    finally:
        session.close()


def get_pipeline(request: Request) -> FacePipeline:
    return request.app.state.pipeline


def get_current_user(
    token: str | None = Cookie(default=None, alias="Token_X"),
    db: Session = Depends(get_db),
) -> User:
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = decode_access_token(token)
        user = db.get(User, int(payload["sub"]))
    except (JWTError, KeyError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if user is None:
        raise HTTPException(status_code=401, detail="User no longer exists")
    return user

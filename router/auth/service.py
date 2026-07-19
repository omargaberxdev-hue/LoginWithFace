import bcrypt
from sqlalchemy import select
from app.models.user import User
from fastapi import HTTPException
from app.utils.security import create_access_token

from sqlalchemy.orm import Session
from .schema import SignUpRequest, SignInRequest, UserResponse


def sign_up(payload: SignUpRequest, db: Session) -> UserResponse:

    existing = db.scalars(select(User).where(User.name == payload.name)).first()

    if existing:
        raise HTTPException(status_code=500, detail="Try anthor Time!")
    
    ImageEmbedding.embed_image
    new_user = User( 
        name=payload.username,
        email=payload.email,
        password=hash_password(payload.password),
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return UserResponse(status=201, Message="Created Succesfully")

# service.py


def sign_in(payload: SignInRequest, db: Session) -> UserResponse:
    existing = db.scalars(select(User).where(User.email == payload.email)).first()

    if not existing or not verify_password(payload.password, existing.password):
        raise HTTPException(status_code=401, detail="invalid credentials")

    return create_access_token({"sub": str(existing.id), "email": existing.email})
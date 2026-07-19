import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session
from fastapi import HTTPException

from app.models.user import User
from app.utils.security import create_access_token
from .schema import SignUpRequest, SignInRequest, UserResponse

from face_detector_class import FaceDetector
from live_detection_single_class import LiveDetection
from image_embedding import ImageEmbedding
from exceptions import LivenessError


def sign_up(payload: SignUpRequest, image: np.ndarray, db: Session) -> UserResponse:

    existing = db.scalars(select(User).where(User.name == payload.name)).first()
    if existing:
        # 409 Conflict is the correct status for "resource already exists",
        # not 500 (500 implies a server bug, this is an expected business case)
        raise HTTPException(status_code=409, detail="Name already taken, try another.")

    # No try/except here: FaceExtractionError, LivenessError, EmbeddingError
    # all propagate up and get converted to proper responses by the
    # handlers registered in error_handlers.py -- this function only
    # needs to express the happy path.

    face_crop = FaceDetector.crop_single_face(image)

    result = LiveDetection.predict(face_crop)
    if result["label"] == "spoof":
        raise LivenessError("Spoof image detected -- sign-up rejected.")

    embedding = ImageEmbedding.embed(face_crop)

    new_user = User(
        name=payload.name,
        age=payload.age,
        embedding=embedding,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return UserResponse(status=201, message="Created successfully")
# service.py


def sign_in(payload: SignInRequest, db: Session) -> UserResponse:
    existing = db.scalars(select(User).where(User.email == payload.email)).first()

    if not existing or not verify_password(payload.password, existing.password):
        raise HTTPException(status_code=401, detail="invalid credentials")

    return create_access_token({"sub": str(existing.id), "email": existing.email})
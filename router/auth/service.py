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

from observability import trace_pipeline, log
from exceptions import LivenessError, IdentityNotFoundError

@trace_pipeline("signup")
def sign_up(payload: SignUpRequest, image: np.ndarray, db: Session) -> UserResponse:
    existing = db.scalars(select(User).where(User.name == payload.name)).first()
    if existing:
        raise HTTPException(status_code=409, detail="Name already taken, try another.")

    face_crop = FaceDetector.crop_single_face(image)

    result = LiveDetection.predict(face_crop)
    if result["label"] == "spoof":
        raise LivenessError("Spoof image detected -- sign-up rejected.")

    embedding = ImageEmbedding.embed_image(face_crop)
    try:
        new_user = User(name=payload.name, age=payload.age, embedding=embedding)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Name already taken, try another.") from exc

    log.info("user_registered", user_id=new_user.id)  # id only -- no name/age/embedding, that's PII
    return new_user


@trace_pipeline("signin")
def sign_in(image: np.ndarray, db: Session) -> str:
    face_crop = FaceDetector.crop_single_face(image)

    result = LiveDetection.predict(face_crop)
    if result["label"] == "spoof":
        raise LivenessError("Spoof image detected -- sign-in rejected.")

    query_embedding = ImageEmbedding.embed_image(face_crop)

    closest_user, distance = db.execute(
        select(User, User.embedding.cosine_distance(query_embedding))
        .order_by(User.embedding.cosine_distance(query_embedding))
        .limit(1)
    ).first()

    if closest_user is None or distance > MATCH_THRESHOLD:
        raise IdentityNotFoundError("No matching registered user found.")

    log.info("user_matched", user_id=closest_user.id, distance=float(distance))
    return create_access_token({"sub": str(closest_user.id)})
 

MATCH_THRESHOLD = 0.30
 
 
def sign_in(image: np.ndarray, db: Session) -> UserResponse:
 
    face_crop = FaceDetector.crop_single_face(image)
 
    result = LiveDetection.predict(face_crop)
    if result["label"] == "spoof":
        raise LivenessError("Spoof image detected -- sign-in rejected.")
 
    query_embedding = ImageEmbedding.embed(face_crop)
 
    closest_user, distance = db.execute(
        select(User, User.embedding.cosine_distance(query_embedding))
        .order_by(User.embedding.cosine_distance(query_embedding))
        .limit(1)
    ).first()
 
    if closest_user is None or distance > MATCH_THRESHOLD:
        raise IdentityNotFoundError("No matching registered user found.")
 
    token = create_access_token({"sub": str(closest_user.id)})
 
    return token
 
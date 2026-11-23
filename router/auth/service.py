import numpy as np
import structlog
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from config.settings import settings
from config.users import User
from Expection.exception import IdentityNotFoundError, UserExistsError
from model.pipeline import FacePipeline
from utils.security import create_access_token

from .schema import SignUpMeta

log = structlog.get_logger()


def sign_up(meta: SignUpMeta, image: np.ndarray, db: Session, pipeline: FacePipeline) -> User:
    # cheap check first, before running three models
    if db.scalar(select(User.id).where(User.name == meta.name)):
        raise UserExistsError("Name already taken, try another.", stage="signup")

    embedding = pipeline.embed_live_face(image, "sign-up")

    user = User(name=meta.name, age=meta.age, embedding=embedding)
    db.add(user)
    try:
        db.commit()
    except IntegrityError as exc:  # lost the race against the unique constraint
        db.rollback()
        raise UserExistsError("Name already taken, try another.", stage="signup") from exc
    db.refresh(user)

    log.info("user_registered", user_id=user.id)  # id only: name/age/embedding are PII
    return user


def sign_in(image: np.ndarray, db: Session, pipeline: FacePipeline) -> str:
    embedding = pipeline.embed_live_face(image, "sign-in")

    distance_expr = User.embedding.cosine_distance(embedding)
    row = db.execute(select(User, distance_expr).order_by(distance_expr).limit(1)).first()

    if row is None:  # empty table (old code crashed unpacking None)
        raise IdentityNotFoundError("No matching registered user found.", stage="match")

    user, distance = row
    if distance > settings.match_threshold:
        log.warning("signin_no_match", best_distance=float(distance))
        raise IdentityNotFoundError("No matching registered user found.", stage="match")

    log.info("user_matched", user_id=user.id, distance=float(distance))
    return create_access_token(str(user.id))

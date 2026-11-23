from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from pydantic import ValidationError
from sqlalchemy.orm import Session

from config.settings import settings
from config.users import User
from depened.dbconnection import get_current_user, get_db, get_pipeline
from model.pipeline import FacePipeline
from utils.images import read_image

from . import service
from .schema import SignUpMeta, UserResponse

router = APIRouter(prefix="/auth", tags=["Auth"])

COOKIE = "Token_X"


# Plain `def` (not async): ML inference + sync SQLAlchemy would block the event loop.
@router.post("/signup", status_code=201, response_model=UserResponse)
def signup_endpoint(
    metadata: str = Form(..., description='JSON: {"name": "...", "age": 25}'),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    pipeline: FacePipeline = Depends(get_pipeline),
):
    try:
        meta = SignUpMeta.model_validate_json(metadata)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.errors(include_url=False, include_context=False, include_input=False),
        )
    return service.sign_up(meta, read_image(image), db, pipeline)


@router.post("/signin")
def signin_endpoint(
    response: Response,
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    pipeline: FacePipeline = Depends(get_pipeline),
):
    token = service.sign_in(read_image(image), db, pipeline)
    response.set_cookie(
        key=COOKIE,
        value=token,
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
        max_age=settings.session_expire_seconds,
    )
    return {"message": "signed in"}


@router.get("/me", response_model=UserResponse)
def me(user: User = Depends(get_current_user)):
    return user


@router.post("/signout")
def signout(response: Response):
    response.delete_cookie(COOKIE)
    return {"message": "signed out"}

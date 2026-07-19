

from fastapi import APIRouter, File, UploadFile, Form

from pydantic import BaseModel

router = ApiRouter(prefix='Auth' ,  dependencies=[Depends(get_db)])


class MetaData(BaseModel): 
    Name: str
    Age: int



@router.post("/signup")
async def sign_up(
    MetaData: str = Form(...),
    image: UploadFile = File(...),
):
  meta_data = MetaData(**json_load(MetaData))
  serice.sign_up(meta_data , image)
  


@router.post("/signin")
async def signin_endpoint(payload: SignInRequest, response: Response, db: Session = Depends(get_db)):
    token = sign_in(payload, db)

    response.set_cookie(
        key="Token_X",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=settings.session_expire_seconds
    )
    return {"message": "signed in"}



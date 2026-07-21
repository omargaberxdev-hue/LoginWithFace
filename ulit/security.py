# security.py
from datetime import datetime, timedelta, timezone
from jose import jwt

SECRET_KEY = "your-secret-key-here" 
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

def create_access_token(data: dict) -> str:
    print("encode secret:", SECRET_KEY)
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(token: str) -> dict:
    print("encode secret:", SECRET_KEY)
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
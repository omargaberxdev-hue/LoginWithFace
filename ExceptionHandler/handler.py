"""
Register these once, in your main FastAPI app setup (e.g. main.py):

    from app.exceptions import FaceExtractionError, LivenessError, EmbeddingError
    from app.error_handlers import register_error_handlers

    app = FastAPI()
    register_error_handlers(app)

After this, any route that lets one of these exceptions propagate
(no try/except needed in the route itself) automatically gets converted
into the correct HTTP response.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from exceptions import FaceExtractionError, LivenessError, EmbeddingError


def register_error_handlers(app: FastAPI):

    @app.exception_handler(FaceExtractionError)
    async def face_extraction_handler(request: Request, exc: FaceExtractionError):
        # user/input error (no face, or too many faces) -- 400, not 500
        return JSONResponse(
            status_code=400,
            content={"error": "face_extraction_failed", "stage": exc.stage, "detail": str(exc)},
        )

    @app.exception_handler(LivenessError)
    async def liveness_handler(request: Request, exc: LivenessError):
        # spoof detected -- this is a rejected request, not a server bug -- 400
        return JSONResponse(
            status_code=400,
            content={"error": "liveness_check_failed", "stage": exc.stage, "detail": str(exc)},
        )

    @app.exception_handler(EmbeddingError)
    async def embedding_handler(request: Request, exc: EmbeddingError):
        # embedding failure is more likely an internal/model issue -- 500
        return JSONResponse(
            status_code=500,
            content={"error": "embedding_failed", "stage": exc.stage, "detail": str(exc)},
        )
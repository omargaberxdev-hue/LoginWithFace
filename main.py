import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from prometheus_client import make_asgi_app

from config import db as database
from config.observeablity import instrument_app, instrument_engine, setup_observability
from config.settings import settings
from ExceptionHandler.handler import register_error_handlers
from router.auth.router import router as auth_router

# >>> check these three class names against your files <<<
from model.Embedding import ImageEmbedder
from model.FaceDetection import FaceDetector
from model.LiveDetection import LivenessDetector
from model.pipeline import FacePipeline

# Providers + logging must exist BEFORE the app is created / instrumented.
setup_observability()
log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    instrument_engine(database.init_engine())

    app.state.pipeline = FacePipeline(
        detector=FaceDetector(),
        liveness=LivenessDetector(),
        embedder=ImageEmbedder(settings.embedder_model_id),
    )
    log.info("startup_complete")
    yield
    log.info("shutdown")


app = FastAPI(title="Login With Face", lifespan=lifespan)

register_error_handlers(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,  # "*" + credentials is rejected by browsers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    structlog.contextvars.clear_contextvars()  # fresh context per request
    structlog.contextvars.bind_contextvars(
        request_id=uuid.uuid4().hex[:12],
        method=request.method,
        path=request.url.path,
    )
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled_exception")
        raise
    log.info(
        "request_completed",
        status_code=response.status_code,
        duration_ms=round((time.perf_counter() - start) * 1000, 1),
    )
    return response


app.include_router(auth_router)
app.mount("/metrics", make_asgi_app())

# Last on purpose: most recently added middleware is outermost, so the OTel server
# span is already active when our middleware logs (=> trace_id in every log line).
instrument_app(app)



@app.get("/")
async def root():
    return {"message": "Login With Face API"}

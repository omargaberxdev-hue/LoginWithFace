import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastapi.openapi.utils import get_openapi

from config.observeablity import setup_observability

@asynccontextmanager
async def lifespan(app: FastAPI):
    ImageEmbedding.load_model(device="cuda" if torch.cuda.is_available() else "cpu")
    init_engine()
    _model = joblib.load("models/liveness_rf.joblib")
    setup_observability()   
    yield



app = FastAPI(lifespan=lifespan)

register_error_handlers(app)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




app.include_router(Logic_router, tags=["Logic"])


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
    )

    schema["openapi"] = "3.0.3"

    # Convert OpenAPI 3.1 file schema to 3.0
    def fix(node):
        if isinstance(node, dict):
            if (
                node.get("type") == "string"
                and node.get("contentMediaType") == "application/octet-stream"
            ):
                node.pop("contentMediaType", None)
                node["format"] = "binary"

            for v in node.values():
                fix(v)

        elif isinstance(node, list):
            for item in node:
                fix(item)

    fix(schema)

    app.openapi_schema = schema
    return schema

app.openapi = custom_openapi


@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}

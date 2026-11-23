"""
Pipeline exceptions. Each carries its own HTTP status + error code, so ONE
handler covers all of them (ExceptionHandler/handler.py).
"""


class PipelineError(Exception):
    status_code = 500
    error = "pipeline_error"

    def __init__(self, message: str = "Internal error", stage: str | None = None):
        self.stage = stage or self.__class__.__name__
        super().__init__(message)


class FaceExtractionError(PipelineError):
    status_code = 400
    error = "face_extraction_failed"


class LivenessError(PipelineError):
    status_code = 400
    error = "liveness_check_failed"


class EmbeddingError(PipelineError):
    status_code = 500
    error = "embedding_failed"


class IdentityNotFoundError(PipelineError):
    status_code = 401
    error = "identity_not_found"


class UserExistsError(PipelineError):
    status_code = 409
    error = "user_exists"

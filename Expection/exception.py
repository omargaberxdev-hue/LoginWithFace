"""
Custom exceptions for the sign-up/sign-in pipeline. Each carries a
`stage` so a single global handler can report exactly which step failed
without needing separate except blocks per stage in the route itself.
"""


class PipelineError(Exception):
    """Common base so a single except/handler can catch all three if needed."""
    def __init__(self, message: str = "Internal Error", stage: str = None):
        self.stage = stage or self.__class__.__name__
        super().__init__(message)


class FaceExtractionError(PipelineError):
    pass


class LivenessError(PipelineError):
    pass


class EmbeddingError(PipelineError):
    pass
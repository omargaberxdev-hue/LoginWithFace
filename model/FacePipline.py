
class FacePipeline:
    """detect -> liveness -> embed. Shared by sign-up and sign-in."""

    def __init__(self, detector, liveness, embedder):
        self.detector = detector
        self.liveness = liveness
        self.embedder = embedder


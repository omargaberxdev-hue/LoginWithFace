import io

import numpy as np
from fastapi import UploadFile
from PIL import Image

from Expection.exception import FaceExtractionError

MAX_BYTES = 5 * 1024 * 1024


def read_image(upload: UploadFile) -> np.ndarray:
    data = upload.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise FaceExtractionError("Image too large (max 5 MB)", stage="decode")
    try:
        with Image.open(io.BytesIO(data)) as img:
            return np.array(img.convert("RGB"))
    except Exception as exc:
        raise FaceExtractionError("Invalid or unsupported image file", stage="decode") from exc

import logging
import os
import urllib.request
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from exceptions import FaceExtractionError

logger = logging.getLogger(__name__)

Box = tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels


# ---- config ------------------------------------------------------------

@dataclass(frozen=True)
class FaceDetectorConfig:
    model_path: str = "blaze_face_short_range.tflite"
    model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
    )
    min_confidence: float = 0.5


# ---- model file management ---------------------------------------------

class ModelDownloader:
    def __init__(self, path: str, url: str):
        self.path = path
        self.url = url

    def ensure(self) -> str:
        if not os.path.exists(self.path):
            logger.info("Downloading face detector model (first run only)...")
            urllib.request.urlretrieve(self.url, self.path)
        return self.path


# ---- detection ---------------------------------------------------------

class FaceDetector:
    """Wraps the MediaPipe detector. Built once, reused for every call."""

    def __init__(self, config: FaceDetectorConfig | None = None):
        self.config = config or FaceDetectorConfig()
        model_path = ModelDownloader(self.config.model_path, self.config.model_url).ensure()

        options = mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            min_detection_confidence=self.config.min_confidence,
        )
        self._detector = mp_vision.FaceDetector.create_from_options(options)

    def detect(self, image_bgr: np.ndarray) -> list[Box]:
        """Returns a list of (x1, y1, x2, y2) boxes in pixel coordinates."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        boxes = []
        result = self._detector.detect(mp_image)
        for detection in result.detections:
            bbox = detection.bounding_box
            x1, y1 = bbox.origin_x, bbox.origin_y
            boxes.append((x1, y1, x1 + bbox.width, y1 + bbox.height))
        return boxes

    def close(self) -> None:
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---- windowing ---------------------------------------------------------
class HannWindow:
    @staticmethod
    def apply(image_gray: np.ndarray) -> np.ndarray:
        if image_gray.ndim != 2:
            raise ValueError(
                f"HannWindow expects a 2D grayscale image, got shape {image_gray.shape}"
            )
        h, w = image_gray.shape
        window_2d = np.outer(np.hanning(h), np.hanning(w))
        return image_gray.astype(np.float32) * window_2d


class FaceCropper:
    def __init__(self, detector: FaceDetector, margin: float = 0.2,
                 window: HannWindow | None = None):
        self.detector = detector
        self.margin = margin
        self.window = window or HannWindow()


    def crop_single_face(self, image_bgr: np.ndarray) -> np.ndarray:
        """Enforces exactly one face. Returns a Hann-windowed grayscale crop (float32)."""
        h, w = image_bgr.shape[:2]
        boxes = self.detector.detect(image_bgr)

        if len(boxes) == 0:
            raise FaceExtractionError("No face detected in the image.")
        if len(boxes) > 1:
            raise FaceExtractionError(
                f"{len(boxes)} faces detected -- exactly one face is required per session."
            )

        x1, y1, x2, y2 = boxes[0]
        bw, bh = x2 - x1, y2 - y1

        mx, my = int(bw * self.margin), int(bh * self.margin)
        x1, y1 = max(0, x1 - mx), max(0, y1 - my)
        x2, y2 = min(w, x2 + mx), min(h, y2 + my)

        crop_bgr = image_bgr[y1:y2, x1:x2]
        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)  # the fix
        return self.window.apply(crop_gray)
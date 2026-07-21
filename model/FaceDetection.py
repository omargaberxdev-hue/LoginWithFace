"""
Single-face detection + crop, with:
  - hard rejection if 0 or >1 faces are detected (a liveness/verification
    session should only ever contain exactly one person)
  - small padding around the tight face box (context helps texture/FFT
    analysis -- moire/halftone evidence can extend slightly beyond the
    jawline)
  - a Hann window applied to the crop before it's handed to FFT analysis,
    to remove the artificial edge discontinuity that causes spectral
    leakage (fake high-frequency energy from the crop boundary itself,
    not from real image content)
"""

import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from observability import trace_stage, log
from exceptions import FaceExtractionError 


class FaceDetector:
    _MODEL_PATH = "blaze_face_short_range.tflite"
    _MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
    )

    @staticmethod
    def _ensure_model_downloaded():
        if not os.path.exists(FaceDetector._MODEL_PATH):
            print("Downloading face detector model (first run only)...")
            urllib.request.urlretrieve(FaceDetector._MODEL_URL, FaceDetector._MODEL_PATH)

    
    @staticmethod
    @trace_stage("face_detect", expected_exceptions=(FaceExtractionError,))
    def detect_faces(image_bgr: np.ndarray, min_confidence: float = 0.5):
        """Returns a list of (x1, y1, x2, y2) boxes in pixel coordinates."""
        FaceDetector._ensure_model_downloaded()

        base_options = mp_python.BaseOptions(model_asset_path=FaceDetector._MODEL_PATH)
        options = mp_vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_confidence,
        )

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        boxes = []
        with mp_vision.FaceDetector.create_from_options(options) as detector:
            result = detector.detect(mp_image)
            for detection in result.detections:
                bbox = detection.bounding_box
                x1, y1 = bbox.origin_x, bbox.origin_y
                x2, y2 = x1 + bbox.width, y1 + bbox.height
                boxes.append((x1, y1, x2, y2))
        return boxes

    @staticmethod
    @trace_stage("face_crop", expected_exceptions=(FaceExtractionError,))
    def crop_single_face(image_bgr: np.ndarray, margin: float = 0.2,
                          min_confidence: float = 0.5) -> np.ndarray:
        """
        Enforces exactly one face. Raises FaceExtractionError otherwise.
        """
        h, w = image_bgr.shape[:2]
        boxes = FaceDetector.detect_faces(image_bgr, min_confidence)

        if len(boxes) == 0:
            raise FaceExtractionError("No face detected in the image.")
        if len(boxes) > 1:
            raise FaceExtractionError(
                f"{len(boxes)} faces detected -- exactly one face is required per session."
            )

        x1, y1, x2, y2 = boxes[0]
        bw, bh = x2 - x1, y2 - y1

        mx, my = int(bw * margin), int(bh * margin)
        x1, y1 = max(0, x1 - mx), max(0, y1 - my)
        x2, y2 = min(w, x2 + mx), min(h, y2 + my)

        return FaceDetector.apply_hann_window(image_bgr[y1:y2, x1:x2])

    @staticmethod
    @trace_stage("hann_window")
    def apply_hann_window(image_gray: np.ndarray) -> np.ndarray:
      
        h, w = image_gray.shape
        hann_row = np.hanning(h)
        hann_col = np.hanning(w)
        window_2d = np.outer(hann_row, hann_col)
        return image_gray.astype(np.float32) * window_2d



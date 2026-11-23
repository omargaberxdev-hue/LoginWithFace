import pathlib
from dataclasses import dataclass
from typing import Iterator, Protocol

import cv2
import joblib
import numpy as np
from skimage.feature import local_binary_pattern

from exceptions import LivenessError


# ---- config ------------------------------------------------------------

@dataclass(frozen=True)
class LivenessConfig:
    grid_size: tuple[int, int] = (6, 6)
    lbp_p: int = 8
    lbp_r: int = 1
    n_bands: int = 4
    model_path: pathlib.Path = pathlib.Path(__file__).parent / "models" / "liveness_rf.joblib"


# ---- shared grid helper ------------------------------------------------

class GridSplitter:
    def __init__(self, grid_size: tuple[int, int]):
        self.grid_size = grid_size

    def cells(self, h: int, w: int) -> Iterator[tuple[int, int, int, int]]:
        rows, cols = self.grid_size
        cell_h, cell_w = h // rows, w // cols
        for i in range(rows):
            for j in range(cols):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                # last row/col absorbs remainder pixels from integer division
                if i == rows - 1:
                    y2 = h
                if j == cols - 1:
                    x2 = w
                yield y1, y2, x1, x2


# ---- feature extractors (common interface) ------------------------------

class FeatureExtractor(Protocol):
    def extract(self, gray: np.ndarray) -> np.ndarray: ...


class LBPExtractor:
    """Branch 1: grid-based uniform LBP histograms."""

    def __init__(self, grid: GridSplitter, P: int, R: int):
        self.grid = grid
        self.P = P
        self.R = R

    def extract(self, gray: np.ndarray) -> np.ndarray:
        lbp = local_binary_pattern(gray, self.P, self.R, method="uniform")
        n_bins = self.P + 2
        h, w = lbp.shape

        hists = []
        for y1, y2, x1, x2 in self.grid.cells(h, w):
            cell = lbp[y1:y2, x1:x2]
            hist, _ = np.histogram(cell.ravel(), bins=n_bins, range=(0, n_bins))
            hist = hist.astype(np.float32)
            hist /= (hist.sum() + 1e-7)
            hists.append(hist)
        return np.concatenate(hists)


class FFTExtractor:
    """Branch 2: block-wise FFT radial band energies."""

    def __init__(self, grid: GridSplitter, n_bands: int):
        self.grid = grid
        self.n_bands = n_bands

    def _band_energies(self, block_gray: np.ndarray) -> np.ndarray:
        f = np.fft.fft2(block_gray.astype(np.float32))
        magnitude = np.log1p(np.abs(np.fft.fftshift(f)))  # log: tame the huge DC term

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        max_radius = radius.max() + 1e-7

        band_edges = np.linspace(0, max_radius, self.n_bands + 1)
        energies = np.zeros(self.n_bands, dtype=np.float32)
        for b in range(self.n_bands):
            mask = (radius >= band_edges[b]) & (radius < band_edges[b + 1])
            energies[b] = magnitude[mask].sum()

        energies /= (energies.sum() + 1e-7)
        return energies


    def extract(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        all_energies = []
        for y1, y2, x1, x2 in self.grid.cells(h, w):
            block = gray[y1:y2, x1:x2]
            if block.shape[0] < 4 or block.shape[1] < 4:  # degenerate edge blocks
                all_energies.append(np.zeros(self.n_bands, dtype=np.float32))
                continue
            all_energies.append(self._band_energies(block))
        return np.concatenate(all_energies)


# ---- the detector (public API) -------------------------------------------

class LivenessDetector:
    def __init__(
        self,
        config: LivenessConfig | None = None,
        extractors: list[FeatureExtractor] | None = None,
        model=None,
    ):
        self.config = config or LivenessConfig()
        grid = GridSplitter(self.config.grid_size)
        self.extractors = extractors or [
            LBPExtractor(grid, self.config.lbp_p, self.config.lbp_r),
            FFTExtractor(grid, self.config.n_bands),
        ]
        self.model = model or joblib.load(self.config.model_path)

    @staticmethod
    def _to_gray(image_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    def extract_features(self, image_bgr: np.ndarray) -> np.ndarray:
        gray = self._to_gray(image_bgr)
        return np.concatenate([e.extract(gray) for e in self.extractors])

    def predict(self, image_bgr: np.ndarray) -> dict:
        try:
            feats = self.extract_features(image_bgr).reshape(1, -1)
        except Exception as e:
            raise LivenessError("Error occurred while processing liveness") from e

        label = self.model.predict(feats)[0]
        proba = self.model.predict_proba(feats)[0][1]  # P(real)

        return {
            "label": "real" if label == 1 else "spoof",
            "confidence": float(proba),
        }
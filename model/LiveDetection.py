

import joblib
import numpy as np
import cv2
from skimage.feature import local_binary_pattern

from observability import trace_stage, log, LIVENESS_RESULT, LIVENESS_CONFIDENCE
from exceptions import LivenessError

import pathlib

class LiveDetection:
    model = None

    GRID_SIZE = (6, 6)   # wider grid -- face fills most of the frame
    LBP_P = 8
    LBP_R = 1
    N_BANDS = 4


    @staticmethod
    def load_model():
        model_path = pathlib.Path(__file__).parent / "models" / "liveness_rf.joblib"
        LiveDetection.model = joblib.load(model_path)
    # ---- shared helpers -----------------------------------------------------

    @staticmethod
    def _to_gray(image_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _iter_cells(h: int, w: int, grid_size):
        rows, cols = grid_size
        cell_h, cell_w = h // rows, w // cols
        for i in range(rows):
            for j in range(cols):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                # last row/col absorbs any remainder pixels from integer division
                if i == rows - 1:
                    y2 = h
                if j == cols - 1:
                    x2 = w
                yield y1, y2, x1, x2

    # ---- Branch 1: grid-based uniform LBP -----------------------------------

    @staticmethod
    @trace_stage("lbp_extract")
    def extract_lbp_grid_histogram(image_bgr: np.ndarray,
                                    grid_size=None, P=None, R=None) -> np.ndarray:
        grid_size = grid_size or LiveDetection.GRID_SIZE
        P = P or LiveDetection.LBP_P
        R = R or LiveDetection.LBP_R

        gray = LiveDetection._to_gray(image_bgr)
        lbp = local_binary_pattern(gray, P, R, method="uniform")
        n_bins = P + 2
        h, w = lbp.shape

        hists = []
        for y1, y2, x1, x2 in LiveDetection._iter_cells(h, w, grid_size):
            cell = lbp[y1:y2, x1:x2]
            hist, _ = np.histogram(cell.ravel(), bins=n_bins, range=(0, n_bins))
            hist = hist.astype(np.float32)
            hist /= (hist.sum() + 1e-7)
            hists.append(hist)

        return np.concatenate(hists)

    # ---- Branch 2: block-wise FFT band energies ------------------------------

    @staticmethod
    def _band_energies(block_gray: np.ndarray, n_bands: int) -> np.ndarray:
        """
        2D FFT of one block -> magnitude spectrum -> radial frequency bands.
        Returns n_bands energy values, normalized to sum to 1 (shape, not
        absolute contrast, so brightness/exposure differences don't dominate).
        """
        f = np.fft.fft2(block_gray.astype(np.float32))
        f_shifted = np.fft.fftshift(f)
        magnitude = np.abs(f_shifted)
        # log-scale: raw FFT magnitude is extremely skewed (huge DC term),
        # log compresses this so band energies aren't dominated by DC alone
        magnitude = np.log1p(magnitude)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        max_radius = radius.max() + 1e-7

        band_edges = np.linspace(0, max_radius, n_bands + 1)
        energies = np.zeros(n_bands, dtype=np.float32)
        for b in range(n_bands):
            mask = (radius >= band_edges[b]) & (radius < band_edges[b + 1])
            energies[b] = magnitude[mask].sum()

        energies /= (energies.sum() + 1e-7)
        return energies

    @staticmethod
    @trace_stage("fft_extract")
    def extract_fft_grid_features(image_bgr: np.ndarray,
                                   grid_size=None, n_bands=None) -> np.ndarray:
        grid_size = grid_size or LiveDetection.GRID_SIZE
        n_bands = n_bands or LiveDetection.N_BANDS

        gray = LiveDetection._to_gray(image_bgr)
        h, w = gray.shape

        all_energies = []
        for y1, y2, x1, x2 in LiveDetection._iter_cells(h, w, grid_size):
            block = gray[y1:y2, x1:x2]
            # skip degenerate tiny blocks at the edges of odd image sizes
            if block.shape[0] < 4 or block.shape[1] < 4:
                all_energies.append(np.zeros(n_bands, dtype=np.float32))
                continue
            all_energies.append(LiveDetection._band_energies(block, n_bands))

        return np.concatenate(all_energies)

    # ---- combined feature vector ---------------------------------------------

    @staticmethod
    @trace_stage("feature_concat")
    def extract_features(image_bgr: np.ndarray) -> np.ndarray:
        lbp_feats = LiveDetection.extract_lbp_grid_histogram(image_bgr)
        fft_feats = LiveDetection.extract_fft_grid_features(image_bgr)
        return np.concatenate([lbp_feats, fft_feats])

    # ---- public inference call -------------------------------------------------

    @staticmethod
    @trace_stage("liveness_predict")
    def predict(image_bgr: np.ndarray) -> dict:
        if LiveDetection.model is None:
            raise RuntimeError("Call LiveDetection.load_model() before predict().")

        # sklearn expects a 2D batch (n_samples, n_features)
        try : 
            feats = LiveDetection.extract_features(image_bgr).reshape(1, -1)
        except:
            raise livenessError("Error Occured while processing Liveness")

        label = LiveDetection.model.predict(feats)[0]
        proba = LiveDetection.model.predict_proba(feats)[0][1]  # P(real)

        return {
            "label": "real" if label == 1 else "spoof",
            "confidence": float(proba),
        }



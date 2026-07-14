"""
feature_extractor.py
--------------------
Extracts EAR, MAR, PERCLOS from MediaPipe FaceMesh.
Each frame is processed independently — no temporal dependency.
"""

import numpy as np
import mediapipe as mp
import cv2
from typing import Optional, Dict

# ── MediaPipe landmark indices ────────────────────────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH_IDX = [78, 308, 13, 14]          # inner-left, inner-right, inner-top, inner-bottom
ALL_EYE   = LEFT_EYE + RIGHT_EYE


class FeatureExtractor:
    """
    MediaPipe FaceMesh wrapper. Computes EAR, MAR, PERCLOS per frame.

    EAR  < 0.20  → eyes likely closed
    MAR  > 0.50  → mouth likely open (yawning)
    PERCLOS > 0.3 → drowsy indicator
    """

    EAR_CLOSED_THRESHOLD = 0.20
    PERCLOS_WINDOW       = 60      # ~2 seconds at 30fps

    def __init__(self):
        self.mp_mesh  = mp.solutions.face_mesh
        self.face_mesh = self.mp_mesh.FaceMesh(
            static_image_mode        = False,
            max_num_faces            = 1,
            refine_landmarks         = True,
            min_detection_confidence = 0.5,
            min_tracking_confidence  = 0.5,
        )
        self._ear_history = []

    # ── Public ────────────────────────────────────────────────────────────────

    def extract(self, frame: np.ndarray) -> Dict:
        """
        Extract all features from a BGR frame.
        Returns dict: EAR, MAR, PERCLOS, face_detected, landmarks, face_crop
        """
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return self._empty()

        lm  = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        pts  = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

        ear  = self._ear(pts)
        mar  = self._mar(pts)
        perc = self._perclos(ear)

        # Crop entire face region for CNN input
        face_crop = self._crop_face(frame, pts)

        return {
            'EAR':          round(float(ear),  4),
            'MAR':          round(float(mar),  4),
            'PERCLOS':      round(float(perc), 4),
            'face_detected': True,
            'landmarks':    pts,
            'face_crop':    face_crop,
        }

    def reset(self):
        """Clear PERCLOS history."""
        self._ear_history.clear()

    def close(self):
        self.face_mesh.close()

    # ── Feature maths ─────────────────────────────────────────────────────────

    def _ear(self, pts: np.ndarray) -> float:
        """Eye Aspect Ratio — average of both eyes."""
        return (self._ear_one(pts, LEFT_EYE) +
                self._ear_one(pts, RIGHT_EYE)) / 2.0

    def _ear_one(self, pts: np.ndarray, idx: list) -> float:
        p = pts[idx]
        v1 = np.linalg.norm(p[1] - p[5])
        v2 = np.linalg.norm(p[2] - p[4])
        h  = np.linalg.norm(p[0] - p[3])
        return float((v1 + v2) / (2.0 * h + 1e-6))

    def _mar(self, pts: np.ndarray) -> float:
        """Mouth Aspect Ratio."""
        p  = pts[MOUTH_IDX]
        v  = np.linalg.norm(p[2] - p[3])    # top-lip to bottom-lip
        h  = np.linalg.norm(p[0] - p[1])    # left-corner to right-corner
        return float(v / (h + 1e-6))

    def _perclos(self, ear: float) -> float:
        """PERCLOS over rolling window."""
        self._ear_history.append(ear)
        if len(self._ear_history) > self.PERCLOS_WINDOW:
            self._ear_history.pop(0)
        closed = sum(1 for e in self._ear_history
                     if e < self.EAR_CLOSED_THRESHOLD)
        return closed / max(len(self._ear_history), 1)

    def _crop_face(self, frame: np.ndarray,
                   pts: np.ndarray, pad: float = 0.25) -> Optional[np.ndarray]:
        """Crop face with padding. Returns None if crop is invalid."""
        h, w   = frame.shape[:2]
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        pw     = (x2 - x1) * pad
        ph     = (y2 - y1) * pad
        x1 = max(0, int(x1 - pw));  y1 = max(0, int(y1 - ph))
        x2 = min(w, int(x2 + pw));  y2 = min(h, int(y2 + ph))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def _empty(self) -> Dict:
        return {'EAR': None, 'MAR': None, 'PERCLOS': None,
                'face_detected': False, 'landmarks': None, 'face_crop': None}

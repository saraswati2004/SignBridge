from dataclasses import dataclass
import base64

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands

MIN_DETECTION_CONFIDENCE = 0.60
MIN_HANDEDNESS_CONFIDENCE = 0.60
MIN_BBOX_WIDTH = 0.12
MIN_BBOX_HEIGHT = 0.12
MIN_BBOX_AREA = 0.02


@dataclass
class HandDetection:
    found: bool
    keypoints: np.ndarray
    bbox: tuple[float, float, float, float]
    handedness_confidence: float

def decode_base64_image(frame_b64: str) -> np.ndarray | None:
    try:
        payload = frame_b64.split(",", 1)[-1]
        image_bytes = base64.b64decode(payload)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

def _empty_detection() -> HandDetection:
    return HandDetection(
        found=False,
        keypoints=np.zeros(63, dtype=np.float32),
        bbox=(0.0, 0.0, 0.0, 0.0),
        handedness_confidence=0.0,
    )

def extract_landmarks(frame):
    if frame is None:
        return None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 🔥 Create fresh instance per request
    with mp_hands.Hands(
        static_image_mode=True,   # ✅ VERY IMPORTANT
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:

        result = hands.process(rgb)

        if not result.multi_hand_landmarks:
            return None

        hand_landmarks = result.multi_hand_landmarks[0]
        landmarks = []

        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        return np.array(landmarks, dtype=np.float32)

def preprocess_landmarks(landmarks):
    """
    shape => (1, 63)
    """
    return landmarks.reshape(1, -1)


def detect_and_extract(frame_bgr: np.ndarray) -> HandDetection:
    if frame_bgr is None:
        return _empty_detection()

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        model_complexity=1,
    ) as hands:
        results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return _empty_detection()

    hand_landmarks = results.multi_hand_landmarks[0]
    xs = np.array([lm.x for lm in hand_landmarks.landmark], dtype=np.float32)
    ys = np.array([lm.y for lm in hand_landmarks.landmark], dtype=np.float32)
    zs = np.array([lm.z for lm in hand_landmarks.landmark], dtype=np.float32)

    min_x = float(xs.min())
    min_y = float(ys.min())
    max_x = float(xs.max())
    max_y = float(ys.max())
    width = max_x - min_x
    height = max_y - min_y
    area = width * height

    handedness_confidence = 0.0
    if results.multi_handedness:
        handedness_confidence = float(results.multi_handedness[0].classification[0].score)

    if handedness_confidence < MIN_HANDEDNESS_CONFIDENCE:
        return _empty_detection()

    if width < MIN_BBOX_WIDTH or height < MIN_BBOX_HEIGHT or area < MIN_BBOX_AREA:
        return _empty_detection()

    keypoints = np.column_stack((xs, ys, zs)).astype(np.float32).flatten()
    return HandDetection(
        found=True,
        keypoints=keypoints,
        bbox=(min_x, min_y, max_x, max_y),
        handedness_confidence=handedness_confidence,
    )

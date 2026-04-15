# for extract data
from function import *
from keras.models import model_from_json
import numpy as np
import cv2

# for fastapi
from fastapi import FastAPI,HTTPException
from pydantic import BaseModels, Field
import logging


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("signbridge")

MODEL_JSON    = "model.json"
MODEL_WEIGHTS = "model_a2i.h5"
MIN_PREDICTION_CONFIDENCE = 0.80
NO_PREDICTION = "-"

app = FastAPI(title="SignBridge API")

# ── Schemas ───────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    frame: str = Field(..., description="Base64 JPEG from browser")

class TopPred(BaseModel):
    letter: str
    conf: float

class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    top5: list[TopPred]
    landmarks_detected: bool

def _legacy_detect_and_extract(frame_bgr: np.ndarray):

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        results = hands.process(rgb)

        frame = cv2.flip(frame, 1)

        # Bigger ROI (hand area)
        x1, y1 = 50, 100
        x2, y2 = 500, 600

        cropframe = frame[y1:y2, x1:x2]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Detect hand
        image, results = mediapipe_detection(cropframe, hands)

        # Draw landmarks
        draw_styled_landmarks(cropframe, results)

        # Extract keypoints
        keypoints = extract_keypoints(results)

@app.get("/")
def root():
    return {"status": "ok", "model_loaded": _S.model is not None,
            "actions": actions.tolist()}
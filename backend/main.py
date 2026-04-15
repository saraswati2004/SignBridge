"""
SignBridge FastAPI Backend — v3
================================
Key fixes vs v2:
  ✅ static_image_mode=True   — correct for per-request (non-streaming) frames
  ✅ No ROI crop              — run MediaPipe on the FULL frame so it can
                                actually locate the hand anywhere in view
  ✅ Re-created per request   — avoids stale tracking state between HTTP calls
  ✅ Lower detection conf     — 0.3 gives MediaPipe more chance to find hand
  ✅ Debug endpoint           — /api/debug/ saves the received frame to disk
                                so you can verify what the server actually sees

Why the old code failed:
  - Tracking mode needs consecutive frames from the SAME video stream.
    Isolated HTTP frames look like random images → MediaPipe loses the hand.
  - ROI crop (50/1280 × small frame) was often just 12px wide → empty crop.
"""

from contextlib import asynccontextmanager
import logging
import os
import mediapipe as mp
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tensorflow.keras.models import model_from_json
from labels import A2I_LABELS
from utils import decode_base64_image, detect_and_extract


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("signbridge")

# ── Actions — paste your exact list from function.py ─────────────────
# Or replace with:  from function import actions; actions = np.array(actions)
actions = np.array(A2I_LABELS)

MODEL_JSON    = "model.json"
MODEL_WEIGHTS = "model_a2i.h5"
MIN_PREDICTION_CONFIDENCE = 0.80
NO_PREDICTION = "-"

# ── App state ─────────────────────────────────────────────────────────
class _S:
    model = None
_S = _S()

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        with open(MODEL_JSON) as f:
            _S.model = model_from_json(f.read())
        _S.model.load_weights(MODEL_WEIGHTS)
        log.info("✅ Model loaded — %d actions", len(actions))
    except FileNotFoundError as e:
        log.warning("⚠️  Model missing (%s) — DEMO mode", e)
    yield

app = FastAPI(title="SignBridge API", version="3.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

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

# ── MediaPipe: one fresh instance per request (static_image_mode) ─────
mp_hands_mod = mp.solutions.hands

def _legacy_detect_and_extract(frame_bgr: np.ndarray):
    """
    Run MediaPipe on a FULL frame (no crop).
    static_image_mode=True is correct for isolated HTTP frames.
    Returns (landmarks_detected: bool, keypoints: np.ndarray shape (63,))
    """
    # MediaPipe needs RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Fresh Hands instance every call — no stale tracking state
    with mp_hands_mod.Hands(
        static_image_mode=True,          # ← correct for API (not live video)
        max_num_hands=1,
        min_detection_confidence=0.3,    # lower = more sensitive
        model_complexity=1,              # 0 is fast but less accurate; 1 is better
    ) as hands:
        results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return False, np.zeros(63, dtype=np.float32)

    lm = results.multi_hand_landmarks[0]
    kp = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32).flatten()
    return True, kp

# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "model_loaded": _S.model is not None,
            "actions": actions.tolist()}

@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": _S.model is not None}

@app.post("/api/predict/", response_model=PredictResponse)
async def predict(req: PredictRequest):
    frame = decode_base64_image(req.frame)
    if frame is None:
        raise HTTPException(400, detail="Bad frame: could not decode image")

    detection = detect_and_extract(frame)
    log.info(
        "Hand detected=%s bbox=%s handedness=%.2f",
        detection.found,
        detection.bbox,
        detection.handedness_confidence,
    )

    if not detection.found:
        return PredictResponse(
            prediction=NO_PREDICTION,
            confidence=0.0,
            top5=[],
            landmarks_detected=False,
        )

    if _S.model is None:
        probs = np.random.dirichlet(np.ones(len(actions)) * 0.3).astype(np.float32)
    else:
        probs = _S.model.predict(detection.keypoints.reshape(1, 63), verbose=0)[0]

    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])
    top5_idx = np.argsort(probs)[::-1][:5]
    top5 = [TopPred(letter=actions[i], conf=round(float(probs[i]), 4)) for i in top5_idx]

    if confidence < MIN_PREDICTION_CONFIDENCE:
        return PredictResponse(
            prediction=NO_PREDICTION,
            confidence=0.0,
            top5=top5,
            landmarks_detected=True,
        )

    return PredictResponse(
        prediction=str(actions[top_idx]),
        confidence=round(confidence, 4),
        top5=top5,
        landmarks_detected=True,
    )

    # 1. Decode frame
    try:
        frame_b64 = req.frame.split(",", 1)[-1]
        raw   = base64.b64decode(frame_b64)
        arr   = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("imdecode returned None")
    except Exception as e:
        raise HTTPException(400, detail=f"Bad frame: {e}")

    log.info("Frame received: %dx%d", frame.shape[1], frame.shape[0])

    # 2. Detect hand + extract keypoints (full frame, no crop)
    found, keypoints = detect_and_extract(frame)
    log.info("Hand detected: %s", found)

    if not found:
        return PredictResponse(
            prediction="–", confidence=0.0,
            top5=[], landmarks_detected=False
        )

    # 3. Inference
    if _S.model is None:
        probs = np.random.dirichlet(np.ones(len(actions)) * 0.3).astype(np.float32)
    else:
        probs = _S.model.predict(keypoints.reshape(1, 63), verbose=0)[0]

    top_idx    = int(np.argmax(probs))
    confidence = float(probs[top_idx])
    top5_idx   = np.argsort(probs)[::-1][:5]
    top5       = [TopPred(letter=actions[i], conf=round(float(probs[i]),4))
                  for i in top5_idx]

    log.info("→ %s (%.0f%%)", actions[top_idx], confidence*100)

    if confidence < MIN_PREDICTION_CONFIDENCE:
        return PredictResponse(
            prediction="â€“",
            confidence=round(confidence, 4),
            top5=top5,
            landmarks_detected=True,
        )

    return PredictResponse(
        prediction=actions[top_idx],
        confidence=round(confidence, 4),
        top5=top5,
        landmarks_detected=True,
    )

# ── DEBUG endpoint: saves received frame to disk so you can inspect it ─
@app.post("/api/debug/")
async def debug(req: PredictRequest):
    """
    POST the same {frame: b64} payload here.
    Saves 'debug_frame.jpg' next to main.py so you can see exactly
    what the server is receiving from the browser.
    Also runs MediaPipe and reports whether hand was found.
    """
    frame = decode_base64_image(req.frame)
    if frame is None:
        raise HTTPException(400, detail="Bad frame: could not decode image")

    debug_path = os.path.abspath("debug_frame.jpg")
    cv2.imwrite(debug_path, frame)

    detection = detect_and_extract(frame)
    return {
        "frame_shape": list(frame.shape),
        "hand_detected": detection.found,
        "bbox": list(detection.bbox),
        "handedness_confidence": round(detection.handedness_confidence, 4),
        "keypoints_nonzero": int(np.count_nonzero(detection.keypoints)),
        "saved_to": debug_path,
    }

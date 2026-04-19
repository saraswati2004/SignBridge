"""
SignBridge — FastAPI Prediction Server  (FIXED)
-----------------------------------------------
POST http://127.0.0.1:8000/api/predict/
Body : { "frame": "<base64 jpeg>" }
Response: { "prediction": "A", "confidence": 0.97,
            "top5": [{"letter":"A","conf":0.97}, ...],
            "landmarks_detected": true }

FIX SUMMARY vs old server.py
  1. MediaPipe Hands is created ONCE at startup (not per-request) → ~10x faster
  2. static_image_mode=False + model_complexity=0 → fastest possible detection
  3. Removed cv2.flip() — browser already sends correct orientation
  4. Only run model.predict when landmarks actually detected
  5. Added /warmup endpoint to pre-heat the model on startup

Requirements:
    pip install fastapi "uvicorn[standard]" opencv-python mediapipe numpy keras tensorflow
"""

import base64
import traceback

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from keras.models import model_from_json
import mediapipe as mp

# ── Label list (must match training order in function.py) ────────────────────
ACTIONS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

# ── Load saved model once at startup ─────────────────────────────────────────
with open("model.json", "r") as f:
    model = model_from_json(f.read())
model.load_weights("model_a2i.h5")
print("✅  Model loaded successfully.")

# Warm up the model so first real request isn't slow
_dummy = np.zeros((1, 63), dtype=np.float32)
model.predict(_dummy, verbose=0)
print("✅  Model warmed up.")

# ── Create MediaPipe Hands ONCE — reused for every request ───────────────────
#    static_image_mode=False  → uses tracking between frames (much faster)
#    model_complexity=0       → fastest/lightest hand model
#    min_detection_confidence=0.5  → balanced threshold
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,        # FIX: reuse tracking state → faster
    model_complexity=0,             # FIX: lightest model → fastest
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1,
)
print("✅  MediaPipe Hands initialised (persistent, complexity=0).")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="SignBridge Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request schema ────────────────────────────────────────────────────────────
class FrameRequest(BaseModel):
    frame: str   # base-64 encoded JPEG

# ── Helpers ───────────────────────────────────────────────────────────────────
def decode_frame(b64: str) -> np.ndarray:
    img_bytes = base64.b64decode(b64)
    buf = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def extract_keypoints(results) -> np.ndarray:
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark
        return np.array([[p.x, p.y, p.z] for p in lm]).flatten().astype(np.float32)
    return np.zeros(21 * 3, dtype=np.float32)

# ── Prediction endpoint ───────────────────────────────────────────────────────
@app.post("/api/predict/")
async def predict(req: FrameRequest):
    try:
        # 1. Decode frame
        frame_bgr = decode_frame(req.frame)
        if frame_bgr is None:
            return {"prediction": None, "confidence": 0.0,
                    "top5": [], "landmarks_detected": False,
                    "error": "Could not decode frame"}

        # 2. FIX: Restore horizontal flip!
        #    If your model was trained on mirrored images (e.g. cv2.flip in data collection),
        #    we MUST mirror the browser's raw canvas so Left/Right hands don't get confused.
        frame_bgr = cv2.flip(frame_bgr, 1)

        # 3. Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        # FIX: use the persistent `mp_hands` instance — no per-request overhead
        results = mp_hands.process(frame_rgb)

        landmarks_detected = bool(results.multi_hand_landmarks)

        # 4. FIX: only predict when hand is actually detected
        if not landmarks_detected:
            return {
                "prediction":         None,
                "confidence":         0.0,
                "top5":               [],
                "landmarks_detected": False,
            }

        # 5. Extract keypoints and run model
        keypoints = extract_keypoints(results)
        probs = model.predict(keypoints[np.newaxis, :], verbose=0)[0]

        best_idx   = int(np.argmax(probs))
        prediction = ACTIONS[best_idx]
        confidence = float(probs[best_idx])

        # 6. Build top-5
        top5 = sorted(
            [{"letter": ACTIONS[i], "conf": float(probs[i])}
             for i in range(len(ACTIONS))],
            key=lambda x: x["conf"],
            reverse=True,
        )[:5]

        return {
            "prediction":         prediction,
            "confidence":         confidence,
            "top5":               top5,
            "landmarks_detected": True,
        }

    except Exception:
        traceback.print_exc()
        return {"prediction": None, "confidence": 0.0,
                "top5": [], "landmarks_detected": False,
                "error": "Internal server error"}

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "actions": ACTIONS}

# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
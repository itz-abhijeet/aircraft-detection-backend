"""
app.py — AirDefenseML Inference API
FastAPI endpoint for real-time military aircraft classification.

Root cause of weights-only .h5:
  The model was saved from Keras 3.10 via model.save_weights() or during
  training callbacks that use weights-only saves. Because there is no
  embedded model config in the HDF5 file, load_model() cannot reconstruct
  the architecture on its own. We rebuild the identical Sequential graph
  here and restore weights by name.

Architecture (reverse-engineered from weight shapes):
  Input (224, 224, 3)
    → EfficientNetB3 (include_top=False, pretrained base weights ignored)
    → GlobalAveragePooling2D
    → BatchNormalization (1536 features)
    → Dense(512, relu)
    → Dropout(0.5)
    → Dense(40, softmax)
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.processor import preprocess_image

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("AirDefenseML")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
MODEL_PATH       = BASE_DIR / "models" / "best_model.h5"
CLASS_INDEX_PATH = BASE_DIR / "models" / "class_indices.json"

# ── Display name map ──────────────────────────────────────────────────────────
DISPLAY_NAMES: dict[str, str] = {
    "A10":        "A-10 Warthog",
    "AH64":       "AH-64 Apache",
    "An72":       "Antonov An-72",
    "B1":         "B-1 Lancer",
    "B2":         "B-2 Spirit",
    "B52":        "B-52 Stratofortress",
    "C130":       "C-130 Hercules",
    "C17":        "C-17 Globemaster",
    "C390":       "KC-390 Millennium",
    "CH53":       "CH-53 Sea Stallion",
    "F14":        "F-14 Tomcat",
    "F15":        "F-15 Eagle",
    "F16":        "F-16 Fighting Falcon",
    "F22":        "F-22 Raptor",
    "F35":        "F-35 Lightning II",
    "F4":         "F-4 Phantom II",
    "Il76":       "Ilyushin Il-76",
    "J10":        "Chengdu J-10",
    "J35":        "Shenyang J-35",
    "J50":        "J-50",
    "JF17":       "JF-17 Thunder",
    "KAAN":       "TAI KAAN",
    "MQ9":        "MQ-9 Reaper",
    "Mi24":       "Mil Mi-24 Hind",
    "Mi26":       "Mil Mi-26",
    "Mi28":       "Mil Mi-28 Havoc",
    "Mi8":        "Mil Mi-8",
    "Mig29":      "MiG-29 Fulcrum",
    "Mig31":      "MiG-31 Foxhound",
    "Mirage2000": "Dassault Mirage 2000",
    "Rafale":     "Dassault Rafale",
    "SR71":       "SR-71 Blackbird",
    "Su24":       "Sukhoi Su-24",
    "Su34":       "Sukhoi Su-34",
    "Su57":       "Sukhoi Su-57",
    "TB2":        "Bayraktar TB2",
    "Tejas":      "HAL Tejas",
    "Tornado":    "Panavia Tornado",
    "UH60":       "UH-60 Black Hawk",
    "Vulcan":     "Avro Vulcan",
}

NUM_CLASSES   = 40
DROPOUT_RATE  = 0.5   # typical fine-tuning value; doesn't affect inference

# ── Shared state ──────────────────────────────────────────────────────────────
state: dict = {}


def build_model() -> tf.keras.Model:
    """
    Load the full model directly from the .h5 file.
    The file was saved with Keras 3.10 and contains the full model config,
    so load_model() can reconstruct the architecture without manual rebuilding.
    """
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ──────────────────────────────────────────────
    log.info("RECON UPLINK :: Loading model from %s …", MODEL_PATH)
    model = build_model()

    # Force a dummy forward pass to warm up XLA / oneDNN
    _ = model.predict(np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0)

    state["model"] = model

    log.info("RECON UPLINK :: Loading class index map …")
    with open(CLASS_INDEX_PATH) as f:
        raw: dict[str, int] = json.load(f)
    state["idx_to_class"] = {v: k for k, v in raw.items()}

    log.info(
        "RECON UPLINK :: System ONLINE. %d classes loaded.",
        len(state["idx_to_class"]),
    )

    yield  # ← application runs

    # ── shutdown ─────────────────────────────────────────────
    log.info("RECON UPLINK :: Shutting down …")
    state.clear()
    tf.keras.backend.clear_session()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AirDefenseML — Aircraft Detection API",
    version="2.0.0",
    description="Real-time military aircraft classification (EfficientNetB3, 40 classes).",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:80",
        "http://localhost",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Response schema ───────────────────────────────────────────────────────────
class TelemetryBlock(BaseModel):
    latency_ms: float
    model: str = "EfficientNetB3"
    input_size: str = "224x224"
    num_classes: int
    vram_usage: str


class DetectResponse(BaseModel):
    status: str
    target: str
    confidence: float
    telemetry: TelemetryBlock


def _vram_label(confidence: float) -> str:
    if confidence >= 0.90:
        return "nominal"
    if confidence >= 0.70:
        return "low"
    return "minimal"


# ── /v1/detect ────────────────────────────────────────────────────────────────
@app.post("/v1/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(..., description="Aerial image for classification")):
    """
    **Target Identification Uplink**

    Upload a `.jpg`, `.png`, or `.webp` image.
    Returns a structured telemetry payload with classification results.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="UPLINK_ERROR: Only image/* content accepted.",
        )

    model: tf.keras.Model = state.get("model")
    idx_to_class: dict    = state.get("idx_to_class")

    if model is None or idx_to_class is None:
        raise HTTPException(
            status_code=503,
            detail="SYSTEM_OFFLINE: Model not yet initialised.",
        )

    raw = await file.read()

    try:
        tensor = preprocess_image(raw)   # → (1, 224, 224, 3) float32
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    t0 = time.perf_counter()
    preds: np.ndarray = model.predict(tensor, verbose=0)  # (1, 40)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    pred_idx    = int(np.argmax(preds[0]))
    confidence  = float(preds[0][pred_idx])
    class_key   = idx_to_class.get(pred_idx, "UNKNOWN")
    target_name = DISPLAY_NAMES.get(class_key, class_key)
    status      = "IDENTIFIED" if confidence >= 0.50 else "UNRESOLVED"

    log.info(
        "SCAN COMPLETE :: target=%s  class_key=%s  conf=%.3f  latency=%.1f ms",
        target_name, class_key, confidence, latency_ms,
    )

    return DetectResponse(
        status=status,
        target=target_name,
        confidence=round(confidence, 4),
        telemetry=TelemetryBlock(
            latency_ms=latency_ms,
            num_classes=len(idx_to_class),
            vram_usage=_vram_label(confidence),
        ),
    )


# ── /v1/detect-url ────────────────────────────────────────────────────────────
class DetectUrlRequest(BaseModel):
    image_url: str

@app.post("/v1/detect-url", response_model=DetectResponse)
async def detect_url(body: DetectUrlRequest):
    """Classify an image from a URL (e.g. Firebase Storage)."""
    import httpx

    model: tf.keras.Model = state.get("model")
    idx_to_class: dict    = state.get("idx_to_class")

    if model is None or idx_to_class is None:
        raise HTTPException(status_code=503, detail="SYSTEM_OFFLINE: Model not yet initialised.")

    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(body.image_url, timeout=15.0)
            r.raise_for_status()
            raw = r.content
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"IMAGE_FETCH_ERROR: {exc}") from exc

    try:
        tensor = preprocess_image(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    t0 = time.perf_counter()
    preds: np.ndarray = model.predict(tensor, verbose=0)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    pred_idx    = int(np.argmax(preds[0]))
    confidence  = float(preds[0][pred_idx])
    class_key   = idx_to_class.get(pred_idx, "UNKNOWN")
    target_name = DISPLAY_NAMES.get(class_key, class_key)
    status      = "IDENTIFIED" if confidence >= 0.50 else "UNRESOLVED"

    log.info("SATELLITE SCAN :: target=%s  conf=%.3f  latency=%.1f ms", target_name, confidence, latency_ms)

    return DetectResponse(
        status=status,
        target=target_name,
        confidence=round(confidence, 4),
        telemetry=TelemetryBlock(
            latency_ms=latency_ms,
            num_classes=len(idx_to_class),
            vram_usage=_vram_label(confidence),
        ),
    )



@app.get("/health", tags=["System"])
async def health():
    """Lightweight uptime probe."""
    return {
        "status": "ONLINE",
        "model_loaded": state.get("model") is not None,
        "classes": len(state.get("idx_to_class", {})),
    }


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")

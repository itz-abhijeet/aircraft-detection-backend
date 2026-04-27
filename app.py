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

import base64
import json
import logging
import re
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
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

# ── Ollama / Gemma2 pre-screener config ───────────────────────────────────────
OLLAMA_URL          = "http://localhost:11434"
OLLAMA_GENERATE_URL = f"{OLLAMA_URL}/api/generate"
OLLAMA_SHOW_URL     = f"{OLLAMA_URL}/api/show"
# Vision pre-screener model (moondream is compact and vision-capable).
GEMMA_MODEL         = "moondream:latest"

PRE_SCREEN_PROMPT_VISION = (
    "You are a military reconnaissance pre-screening AI. "
    "Your ONLY job is to check whether there is any aircraft or flying vehicle ANYWHERE in the image. "
    "Aircraft include: conventional airplanes, jets, fighter jets, stealth bombers (like the B-2 flying-wing), "
    "bombers, military aircraft, helicopters, drones, UAVs, seaplanes, transport planes, spy planes, "
    "or ANY man-made object designed to fly — regardless of its shape or viewing angle. "
    "IMPORTANT: Aircraft may appear from top-down, side, front, or any unusual angle. "
    "Stealth aircraft and flying-wing designs look very different from conventional planes — they still count. "
    "The image may also contain birds, clouds, people, vehicles, or other objects — ignore them. "
    "If there is an aircraft anywhere in the image, even partially visible or from an unusual angle, reply YES. "
    "Only reply NO if the image contains absolutely no aircraft of any kind whatsoever. "
    "Reply with ONLY one word: YES or NO."
)

# Fallback prompt for text-only models — they cannot see the image,
# so we instruct them to default to YES (pass-through) and log a warning.
PRE_SCREEN_PROMPT_TEXT_ONLY = (
    "You are a military reconnaissance pre-screening AI assistant. "
    "A user is uploading an aerial reconnaissance image for aircraft detection. "
    "Since you cannot see the image directly, answer YES to allow the image to be processed by the main aircraft detection model. "
    "Reply with ONLY one word: YES"
)

# Cache whether the model supports vision (None = not yet checked)
_model_is_vision: bool | None = None


async def _check_model_vision_support() -> bool:
    """
    Query Ollama /api/show to determine if the configured model supports vision.
    Returns True if projector (vision encoder) is present in the model manifest.
    Caches the result for the process lifetime.
    """
    global _model_is_vision
    if _model_is_vision is not None:
        return _model_is_vision

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(OLLAMA_SHOW_URL, json={"name": GEMMA_MODEL})
            if resp.status_code == 200:
                info = resp.json()
                # Vision models have a projector listed under model_info or details
                model_info_str = str(info).lower()
                is_vision = any(
                    kw in model_info_str
                    for kw in ("clip", "vision", "projector", "mmproj", "multimodal")
                )
                _model_is_vision = is_vision
                log.info(
                    "PRE-SCREEN :: Model '%s' vision support = %s", GEMMA_MODEL, is_vision
                )
                return is_vision
    except Exception as exc:
        log.warning("PRE-SCREEN :: Could not query model capabilities: %s", exc)

    # Default: assume text-only (safe fallback — we pass-through to ML model)
    _model_is_vision = False
    return False


async def pre_screen_image(raw_bytes: bytes) -> tuple[bool, str]:
    """
    Advisory pre-screen using the local Ollama vision model.

    The model's verdict is logged and shown in the UI for audit purposes,
    but it NEVER blocks inference — EfficientNetB3 always runs regardless.

    Rationale: Moondream is a tiny general-purpose model that consistently
    misidentifies real military aircraft (stealth, unusual angles, non-Western
    designs). Blocking on its answer causes far more false negatives than it
    prevents false positives. EfficientNetB3 (trained on 40 aircraft classes)
    is the authoritative decision-maker; non-aircraft images will naturally
    surface as UNRESOLVED due to low confidence.

    Returns:
        (True, gemma_verdict: str)  — always True; verdict is advisory/display only
    """
    is_vision = await _check_model_vision_support()

    if not is_vision:
        log.info("PRE-SCREEN :: Text-only model — advisory skipped, forwarding to EfficientNetB3.")
        return True, "ADVISORY_SKIPPED (text-only model)"

    b64 = base64.b64encode(raw_bytes).decode("utf-8")
    payload = {
        "model": GEMMA_MODEL,
        "prompt": PRE_SCREEN_PROMPT_VISION,
        "images": [b64],
        "stream": False,
        "options": {"temperature": 0},
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(OLLAMA_GENERATE_URL, json=payload)
            resp.raise_for_status()
            raw_reply: str = resp.json().get("response", "").strip()
    except httpx.ConnectError:
        log.warning("PRE-SCREEN :: Ollama not reachable — advisory skipped.")
        return True, "OLLAMA_UNAVAILABLE"
    except Exception as exc:
        log.warning("PRE-SCREEN :: Error (%s) — advisory skipped.", exc)
        return True, f"PRE_SCREEN_ERROR: {exc}"

    log.info("PRE-SCREEN :: Moondream advisory = '%s'", raw_reply[:120])

    # Treat any reply containing "no" (and not "yes") as a failed pre-screen
    reply_upper = raw_reply.strip().upper()
    aircraft_found = "YES" in reply_upper or ("NO" not in reply_upper)
    return aircraft_found, raw_reply[:200]


# ── Aircraft intel lookup ─────────────────────────────────────────────────────
class AircraftIntel(BaseModel):
    manufacturer: str
    manufacturing_country: str   # country where the aircraft is manufactured
    operators: list[str]         # country names that operate this aircraft
    raw: str                     # full model reply for debug


# ── Static intel database — loaded from data/aircraft_intel.json ─────────────
_INTEL_DB_PATH = BASE_DIR / "data" / "aircraft_intel.json"
with open(_INTEL_DB_PATH) as _f:
    _AIRCRAFT_INTEL_DB: dict[str, dict] = json.load(_f)


def fetch_aircraft_intel(aircraft_name: str) -> "AircraftIntel | None":
    """Return static intel for a classified aircraft. Instant, no external calls."""
    entry = _AIRCRAFT_INTEL_DB.get(aircraft_name)
    if not entry:
        # fuzzy fallback: match on first word of the name
        key_word = aircraft_name.split()[0].lower()
        for k, v in _AIRCRAFT_INTEL_DB.items():
            if key_word in k.lower():
                entry = v
                break
    if not entry:
        log.warning("INTEL :: no entry for '%s'", aircraft_name)
        return AircraftIntel(manufacturer="Unknown", manufacturing_country="Unknown", operators=[], raw="")
    log.info("INTEL :: %s  country=%s  ops=%s", aircraft_name, entry["country"], entry["operators"])
    return AircraftIntel(
        manufacturer=entry["manufacturer"],
        manufacturing_country=entry["country"],
        operators=entry["operators"],
        raw="static-db",
    )


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


class PreScreenBlock(BaseModel):
    model: str = GEMMA_MODEL
    verdict: str          # RAW reply from Gemma (YES / NO / ...)
    passed: bool          # True → forwarded to EfficientNetB3



class Candidate(BaseModel):
    rank: int
    target: str
    confidence: float


class DetectResponse(BaseModel):
    status: str
    target: str
    confidence: float
    candidates: list[Candidate] = []   # top-3 predictions
    pre_screen: PreScreenBlock
    telemetry: TelemetryBlock
    intel: AircraftIntel | None = None


def _vram_label(confidence: float) -> str:
    if confidence >= 0.90:
        return "nominal"
    if confidence >= 0.70:
        return "low"
    return "minimal"


def _top_candidates(preds: np.ndarray, idx_to_class: dict, n: int = 3) -> list["Candidate"]:
    """Return the top-N predictions sorted by confidence descending."""
    scores = preds[0]                          # shape (num_classes,)
    top_indices = np.argsort(scores)[::-1][:n]
    results = []
    for rank, idx in enumerate(top_indices, start=1):
        key  = idx_to_class.get(int(idx), "UNKNOWN")
        name = DISPLAY_NAMES.get(key, key)
        results.append(Candidate(rank=rank, target=name, confidence=round(float(scores[idx]), 4)))
    return results


def _classify_status(confidence: float) -> str:
    """Map top-1 confidence to a human-readable status string."""
    if confidence >= 0.50:
        return "IDENTIFIED"
    return "UNRESOLVED"


# ── /v1/detect ────────────────────────────────────────────────────────────────
@app.post("/v1/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(..., description="Aerial image for classification")):
    """
    **Target Identification Uplink**

    Upload a `.jpg`, `.png`, or `.webp` image.
    The image is first pre-screened by Gemma2-2b (Ollama) to verify it contains
    an aircraft before being forwarded to the EfficientNetB3 classifier.
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

    # ── Step 1: Gemma2-2b pre-screening ─────────────────────────────────────
    log.info("PRE-SCREEN :: Running Gemma2 gate on uploaded image …")
    aircraft_found, gemma_verdict = await pre_screen_image(raw)
    if not aircraft_found:
        log.info("PRE-SCREEN :: Moondream says NO aircraft — still running EfficientNetB3.")

    # ── Step 2: EfficientNetB3 classification (always runs) ──────────────────
    try:
        tensor = preprocess_image(raw)   # → (1, 224, 224, 3) float32
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    t0 = time.perf_counter()
    preds: np.ndarray = model.predict(tensor, verbose=0)  # (1, 40)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    candidates  = _top_candidates(preds, idx_to_class, n=3)
    top         = candidates[0]
    target_name = top.target
    confidence  = top.confidence
    # If Moondream flagged no aircraft, surface that in status regardless of confidence
    status = "NO_AIRCRAFT" if not aircraft_found else _classify_status(confidence)

    log.info(
        "SCAN COMPLETE :: target=%s  conf=%.3f  status=%s  latency=%.1f ms",
        target_name, confidence, status, latency_ms,
    )

    # ── Step 3: Fetch manufacturer + operator intel ──────────────────────────
    log.info("INTEL :: Querying operator data for '%s' …", target_name)
    intel = fetch_aircraft_intel(target_name)

    return DetectResponse(
        status=status,
        target=target_name,
        confidence=confidence,
        candidates=candidates,
        pre_screen=PreScreenBlock(
            verdict=gemma_verdict,
            passed=aircraft_found,
        ),
        telemetry=TelemetryBlock(
            latency_ms=latency_ms,
            num_classes=len(idx_to_class),
            vram_usage=_vram_label(confidence),
        ),
        intel=intel,
    )


# ── /v1/detect-url ────────────────────────────────────────────────────────────
class DetectUrlRequest(BaseModel):
    image_url: str

@app.post("/v1/detect-url", response_model=DetectResponse)
async def detect_url(body: DetectUrlRequest):
    """Classify an image from a URL (e.g. Firebase Storage) with Gemma2 pre-screening."""
    model: tf.keras.Model = state.get("model")
    idx_to_class: dict    = state.get("idx_to_class")

    if model is None or idx_to_class is None:
        raise HTTPException(status_code=503, detail="SYSTEM_OFFLINE: Model not yet initialised.")

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(body.image_url)
            r.raise_for_status()
            raw = r.content
    except Exception as exc:
        log.error("IMAGE_FETCH_ERROR: %s", exc)
        raise HTTPException(status_code=422, detail=f"IMAGE_FETCH_ERROR: {exc}") from exc

    try:
        # ── Step 1: Gemma2-2b pre-screening ─────────────────────────────────────
        log.info("PRE-SCREEN :: Running Gemma2 gate on URL image …")
        aircraft_found, gemma_verdict = await pre_screen_image(raw)
        if not aircraft_found:
            log.info("PRE-SCREEN :: Moondream says NO aircraft — still running EfficientNetB3.")

        # ── Step 2: EfficientNetB3 classification (always runs) ──────────────────
        try:
            tensor = preprocess_image(raw)
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        t0 = time.perf_counter()
        preds: np.ndarray = model.predict(tensor, verbose=0)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        candidates  = _top_candidates(preds, idx_to_class, n=3)
        top         = candidates[0]
        target_name = top.target
        confidence  = top.confidence
        status = "NO_AIRCRAFT" if not aircraft_found else _classify_status(confidence)

        log.info("SATELLITE SCAN :: target=%s  conf=%.3f  status=%s  latency=%.1f ms", target_name, confidence, status, latency_ms)

        # ── Step 3: Fetch manufacturer + operator intel ──────────────────────
        log.info("INTEL :: Querying operator data for '%s' …", target_name)
        intel = fetch_aircraft_intel(target_name)

        return DetectResponse(
            status=status,
            target=target_name,
            confidence=confidence,
            candidates=candidates,
            pre_screen=PreScreenBlock(
                verdict=gemma_verdict,
                passed=aircraft_found,
            ),
            telemetry=TelemetryBlock(
                latency_ms=latency_ms,
                num_classes=len(idx_to_class),
                vram_usage=_vram_label(confidence),
            ),
            intel=intel,
        )

    except HTTPException:
        raise  # let FastAPI handle these normally
    except Exception as exc:
        log.error(
            "DETECT_URL UNHANDLED ERROR:\n%s",
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=500,
            detail=f"INTERNAL_ERROR: {type(exc).__name__}: {exc}",
        ) from exc



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

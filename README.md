# AirDefenseML — Military Aircraft Detection System

Real-time military aircraft classification using a fine-tuned **EfficientNetB3** model.
Upload an aerial image (or stream live tracking events from Firebase) and receive an instant identification across **40 aircraft classes** — from the F-22 Raptor to the Bayraktar TB2 — complete with manufacturer intel and operator nations.

---

## Features

- 🛩️ **40-class classifier** — EfficientNetB3 fine-tuned on military aviation imagery
- 🔍 **Moondream advisory pre-screen** — compact vision model provides a YES/NO advisory; EfficientNetB3 **always runs** regardless of the advisory verdict
- 📡 **Live Satellite Feed mode** — Firebase Realtime Database listener auto-classifies new tracking events and plots targets on a live tactical map (Leaflet.js)
- 🧠 **Static Intel Brief** — deterministic, hardcoded database returns manufacturer, manufacturing country, and operator nations for every classified aircraft instantly (no LLM calls)
- ⚡ **Top-3 candidates** — every response includes the top-3 predictions ranked by confidence
- 🐳 **Docker-ready** — single `docker-compose up --build` spins up both services

---

## Project Structure

```
aircraft-detection-backend/
├── app.py                    # FastAPI backend — inference API
├── utils/
│   └── processor.py          # Image preprocessing pipeline
├── data/
│   └── aircraft_intel.json   # Static manufacturer / operator database (40 entries)
├── models/                   # ⚠️  Not included — see below
│   ├── best_model.h5
│   └── class_indices.json
├── frontend/                 # React 19 + Vite + Tailwind CSS
│   └── src/
│       ├── App.jsx           # Manual Scan mode
│       ├── LiveTracker.jsx   # Live Satellite Feed mode (Firebase + Leaflet)
│       └── firebase.js       # Firebase SDK config
├── requirements.txt
├── docker-compose.yml
└── Dockerfile.backend
```

---

## ⚠️ Model Files Required

The model weights are not included in this repository due to size constraints.
Download them and place them in the `models/` folder before running.

📥 **[Download Model Files from Google Drive](https://drive.google.com/drive/folders/1_BB49rOzmIKf6JYUcp4vkPbPMdshq0IN?usp=sharing)**

| File | Size | Description |
|---|---|---|
| `best_model.h5` | ~133 MB | Full Keras 3.10 model (EfficientNetB3 + classification head) |
| `class_indices.json` | < 1 KB | `{"A10": 0, "AH64": 1, ..., "Vulcan": 39}` |

---

## Running Locally

### Prerequisites

| Requirement | Version |
|---|---|
| Python | **3.12** (TensorFlow does not support 3.13+) |
| Node.js | 18+ |
| Ollama (optional) | Latest — for Moondream advisory pre-screening |

### 1. Backend

```bash
# Install dependencies
py -3.12 -m pip install -r requirements.txt

# Start the FastAPI server
py -3.12 app.py
```

Backend runs at **`http://localhost:8000`**

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at **`http://localhost:5173`** (or `5174` if `5173` is busy)

Open the URL, select a mode (**Manual Scan** or **Live Satellite Feed**), upload an image, and hit **Execute Scan**.

### 3. Moondream (optional)

The Moondream pre-screener is **advisory only** — EfficientNetB3 always runs even if Ollama is unreachable or the model says NO.

```bash
# Install Ollama, then pull the model
ollama pull moondream
```

If Ollama is not running, the backend logs `OLLAMA_UNAVAILABLE` and continues normally.

---

## Running with Docker

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| Frontend | `http://localhost` |
| Backend API | `http://localhost:8000` |

> Make sure the `models/` folder is populated before building the image.

---

## API Reference

### `POST /v1/detect`
Upload an image file for manual classification.

**Request:** `multipart/form-data` with field `file` (`.jpg`, `.png`, or `.webp`)

**Response:**
```json
{
  "status": "IDENTIFIED",
  "target": "F-22 Raptor",
  "confidence": 0.9741,
  "candidates": [
    { "rank": 1, "target": "F-22 Raptor",     "confidence": 0.9741 },
    { "rank": 2, "target": "F-35 Lightning II","confidence": 0.0183 },
    { "rank": 3, "target": "SR-71 Blackbird",  "confidence": 0.0041 }
  ],
  "pre_screen": {
    "model": "moondream:latest",
    "verdict": "YES",
    "passed": true
  },
  "telemetry": {
    "latency_ms": 312.4,
    "model": "EfficientNetB3",
    "input_size": "224x224",
    "num_classes": 40,
    "vram_usage": "nominal"
  },
  "intel": {
    "manufacturer": "Lockheed Martin",
    "manufacturing_country": "United States",
    "operators": ["United States"],
    "raw": "static-db"
  }
}
```

**Status codes:**

| `status` | Meaning |
|---|---|
| `IDENTIFIED` | Top-1 confidence ≥ 50% |
| `UNRESOLVED` | Top-1 confidence < 50% |
| `NO_AIRCRAFT` | Moondream advisory said NO (EfficientNetB3 still ran) |

---

### `POST /v1/detect-url`
Classify an image from a URL (used by the Live Satellite Feed / Firebase integration).

**Request body:**
```json
{ "image_url": "https://storage.googleapis.com/..." }
```

**Response:** Same schema as `/v1/detect`.

---

### `GET /health`
Lightweight uptime probe.

```json
{ "status": "ONLINE", "model_loaded": true, "classes": 40 }
```

---

## Inference Pipeline

```
Uploaded Image
      │
      ▼
[Moondream:latest] ─── Advisory only (YES / NO) ─── logged + shown in UI
      │                 Does NOT block inference
      ▼
[EfficientNetB3]  ─── Always runs ───────────────── Top-3 predictions
      │
      ▼
[Static Intel DB] ─── Instant lookup ───────────── Manufacturer + operators
      │
      ▼
DetectResponse JSON
```

**Image preprocessing** (`utils/processor.py`):
1. Decode bytes → PIL Image
2. Convert to RGB (handles RGBA, grayscale, palette modes)
3. EXIF auto-rotate (fixes mobile camera photos)
4. Resize to **224 × 224** (Lanczos)
5. Cast to `float32` — EfficientNetB3's internal preprocessing layer handles normalisation

---

## Live Satellite Feed (Firebase Integration)

The **Live Satellite Feed** mode listens to the `plane_trackings` node in Firebase Realtime Database. When a new tracking event arrives, it:

1. Reads `estimatedPlaneLat`, `estimatedPlaneLon`, `heading`, and `imageUrl` from the event payload
2. Calls `POST /v1/detect-url` with the image URL
3. Plots the target on a Leaflet tactical map (dark CartoDB tile layer)
4. Displays the classification result, confidence, Intel Brief, and telemetry in the analysis panel

**Required Firebase fields per tracking document:**

| Field | Type | Description |
|---|---|---|
| `estimatedPlaneLat` | `number` | Latitude (decimal degrees) |
| `estimatedPlaneLon` | `number` | Longitude (decimal degrees) |
| `heading` | `number` | Heading in degrees |
| `imageUrl` | `string` | Publicly accessible image URL |
| `timestamp` | `number` | Unix timestamp (ms) |

---

## Supported Aircraft Classes

| Key | Display Name |
|---|---|
| A10 | A-10 Warthog |
| AH64 | AH-64 Apache |
| An72 | Antonov An-72 |
| B1 | B-1 Lancer |
| B2 | B-2 Spirit |
| B52 | B-52 Stratofortress |
| C130 | C-130 Hercules |
| C17 | C-17 Globemaster |
| C390 | KC-390 Millennium |
| CH53 | CH-53 Sea Stallion |
| F14 | F-14 Tomcat |
| F15 | F-15 Eagle |
| F16 | F-16 Fighting Falcon |
| F22 | F-22 Raptor |
| F35 | F-35 Lightning II |
| F4 | F-4 Phantom II |
| Il76 | Ilyushin Il-76 |
| J10 | Chengdu J-10 |
| J35 | Shenyang J-35 |
| J50 | J-50 |
| JF17 | JF-17 Thunder |
| KAAN | TAI KAAN |
| MQ9 | MQ-9 Reaper |
| Mi24 | Mil Mi-24 Hind |
| Mi26 | Mil Mi-26 |
| Mi28 | Mil Mi-28 Havoc |
| Mi8 | Mil Mi-8 |
| Mig29 | MiG-29 Fulcrum |
| Mig31 | MiG-31 Foxhound |
| Mirage2000 | Dassault Mirage 2000 |
| Rafale | Dassault Rafale |
| SR71 | SR-71 Blackbird |
| Su24 | Sukhoi Su-24 |
| Su34 | Sukhoi Su-34 |
| Su57 | Sukhoi Su-57 |
| TB2 | Bayraktar TB2 |
| Tejas | HAL Tejas |
| Tornado | Panavia Tornado |
| UH60 | UH-60 Black Hawk |
| Vulcan | Avro Vulcan |

---

## Tech Stack

| Layer | Stack |
|---|---|
| Backend | FastAPI · TensorFlow ≥ 2.16 · Keras 3.10 · EfficientNetB3 · Pillow · httpx |
| Pre-screener | Ollama · Moondream (vision-capable, advisory only) |
| Frontend | React 19 · Vite · Tailwind CSS · Lucide React |
| Live Tracking | Firebase Realtime Database · Leaflet.js (react-leaflet) |
| Containerisation | Docker · Docker Compose |

# AirDefenseML — Aircraft Detection System

Real-time military aircraft classification using EfficientNetB3. Upload an aerial image and get an instant identification from 40 aircraft classes (F-22, MiG-29, B-2, Rafale, etc.).

---

## Project Structure

```
aircraft-detection-backend/
├── app.py                  # FastAPI backend
├── utils/processor.py      # Image preprocessing
├── models/                 # ⚠️ Not included in repo — see below
│   ├── best_model.h5
│   └── class_indices.json
├── frontend/               # React + Vite frontend
├── requirements.txt
└── docker-compose.yml
```

---

## ⚠️ Model Files Required

The model files are not included in this repository due to size limits. Download them from the link below and place them in the `models/` folder.

📥 **[Download Model Files from Google Drive](https://drive.google.com/drive/folders/1_BB49rOzmIKf6JYUcp4vkPbPMdshq0IN?usp=sharing)**

Place the following files in the `models/` folder:

```
models/
├── best_model.h5        (133 MB — EfficientNetB3 trained weights)
└── class_indices.json   (class name → index mapping)
```

`class_indices.json` format:
```json
{"A10": 0, "AH64": 1, "An72": 2, ..., "Vulcan": 39}
```

---

## Running Locally

### Prerequisites
- Python 3.12 (TensorFlow does not support 3.13+)
- Node.js 18+

### 1. Backend

```bash
# Install dependencies
py -3.12 -m pip install -r requirements.txt

# Start the API server
py -3.12 app.py
```

Backend runs at `http://localhost:8000`

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173` (or 5174 if 5173 is busy)

Open the URL in your browser, upload an aircraft image, and hit **Execute Scan**.

---

## Running with Docker

```bash
docker-compose up --build
```

- Frontend → `http://localhost`
- Backend API → `http://localhost:8000`

> Make sure the `models/` folder is populated before building.

---

## Supported Aircraft Classes

A-10 Warthog, AH-64 Apache, Antonov An-72, B-1 Lancer, B-2 Spirit, B-52 Stratofortress, C-130 Hercules, C-17 Globemaster, KC-390, CH-53 Sea Stallion, F-14 Tomcat, F-15 Eagle, F-16 Fighting Falcon, F-22 Raptor, F-35 Lightning II, F-4 Phantom II, Ilyushin Il-76, Chengdu J-10, Shenyang J-35, J-50, JF-17 Thunder, TAI KAAN, MQ-9 Reaper, Mil Mi-24, Mil Mi-26, Mil Mi-28, Mil Mi-8, MiG-29, MiG-31, Mirage 2000, Dassault Rafale, SR-71 Blackbird, Su-24, Su-34, Su-57, Bayraktar TB2, HAL Tejas, Panavia Tornado, UH-60 Black Hawk, Avro Vulcan

---

## Tech Stack

- Backend: FastAPI + TensorFlow 2.21 + Keras 3.10 + EfficientNetB3
- Frontend: React 19 + Vite + Tailwind CSS

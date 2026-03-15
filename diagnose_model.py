"""
diagnose_model.py  — Diagnostic script to identify model loading issues.
Run: .\venv\Scripts\python.exe diagnose_model.py
"""
import sys
print(f"Python: {sys.version}")

import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")

try:
    import keras
    print(f"Keras: {keras.__version__}")
except ImportError:
    pass

import json, traceback
from pathlib import Path

MODEL_PATH = Path("models/best_model.h5")
CLASS_PATH = Path("models/class_indices.json")

print(f"\nModel file exists: {MODEL_PATH.exists()}")
print(f"Class index file exists: {CLASS_PATH.exists()}")

# Try load with compile=False first
print("\n--- Attempting tf.keras.models.load_model(compile=False) ---")
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"SUCCESS! Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Num layers: {len(model.layers)}")
except Exception as e:
    traceback.print_exc()

# Try with safe_mode disabled (Keras 3 needs this sometimes for legacy models)
print("\n--- Attempting with safe_mode=False ---")
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
    print(f"SUCCESS (safe_mode=False)! Input shape: {model.input_shape}")
except Exception as e:
    traceback.print_exc()

# Try h5py raw read to get model config
print("\n--- Inspecting HDF5 structure with h5py ---")
try:
    import h5py
    with h5py.File(MODEL_PATH, "r") as f:
        print(f"Top-level keys: {list(f.keys())}")
        if "model_config" in f.attrs:
            import json
            cfg = json.loads(f.attrs["model_config"])
            print(f"Model class_name: {cfg.get('class_name')}")
        if "keras_version" in f.attrs:
            print(f"Saved with Keras version: {f.attrs['keras_version']}")
        if "backend" in f.attrs:
            print(f"Saved with backend: {f.attrs['backend']}")
except Exception as e:
    traceback.print_exc()

print("\n--- Class indices ---")
with open(CLASS_PATH) as f:
    classes = json.load(f)
print(f"Total classes: {len(classes)}")
print(f"Sample: {dict(list(classes.items())[:5])}")

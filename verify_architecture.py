"""
verify_architecture.py — Confirm EfficientNetB3 matches the weight shapes.
"""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF noise

import tensorflow as tf
import h5py
from pathlib import Path

MODEL_PATH = Path("models/best_model.h5")

# Confirm final Dense layer output size
with h5py.File(MODEL_PATH, "r") as f:
    top = f["model_weights"]["top_level_model_weights"]
    for k in top.keys():
        grp = top[k]
        if hasattr(grp, "shape"):
            print(f"  top_level {k}: {grp.shape}")
        else:
            for sub in grp.keys():
                ds = grp[sub]
                if hasattr(ds, "shape"):
                    print(f"  top_level {k}/{sub}: {ds.shape}")

"""
probe_weights.py — List all weight layer names so we can reconstruct architecture.
"""
import h5py
from pathlib import Path

MODEL_PATH = Path("models/best_model.h5")

def print_keys(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"  DATASET: {name:80s} shape={obj.shape}")

with h5py.File(MODEL_PATH, "r") as f:
    print("=== TOP-LEVEL ATTRS ===")
    for k, v in f.attrs.items():
        print(f"  {k}: {v[:200] if isinstance(v, str) else v}")

    print("\n=== TOP-LEVEL KEYS ===")
    print(list(f.keys()))

    if "model_weights" in f:
        print("\n=== model_weights LAYER NAMES ===")
        for layer_name in f["model_weights"].keys():
            print(f"  {layer_name}")
            grp = f["model_weights"][layer_name]
            for sub in grp.keys():
                sub_grp = grp[sub]
                if hasattr(sub_grp, 'shape'):
                    print(f"    → {sub}: {sub_grp.shape}")
                else:
                    for w in sub_grp.keys():
                        ds = sub_grp[w]
                        if hasattr(ds, 'shape'):
                            print(f"    → {sub}/{w}: {ds.shape}")

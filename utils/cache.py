from pathlib import Path
import pickle

def save_cache(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_cache(path: Path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

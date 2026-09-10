import json
from pathlib import Path
def save_threshold(v,path='models/threshold.json'):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps({'threshold':float(v)},indent=2))
def load_threshold(path='models/threshold.json',default=.5):
    p=Path(path); return float(json.loads(p.read_text())['threshold']) if p.exists() else default

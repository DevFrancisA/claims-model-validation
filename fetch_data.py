"""Download freMTPL2 frequency + severity tables from OpenML into data/."""
from pathlib import Path
from sklearn.datasets import fetch_openml

d = Path(__file__).parent / "data"
d.mkdir(exist_ok=True)
fetch_openml(data_id=41214, as_frame=True, parser="auto").frame.to_csv(d / "freMTPL2freq.csv", index=False)
fetch_openml(data_id=41215, as_frame=True, parser="auto").frame.to_csv(d / "freMTPL2sev.csv", index=False)
print("data ready")

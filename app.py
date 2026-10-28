"""Local UI server (stdlib only). Run:  python app.py   then open http://localhost:8000"""
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import gbm_features, glm_features

ROOT = Path(__file__).resolve().parent
UI, OUT = ROOT / "ui", ROOT / "outputs"
MODELS = None


def models():
    global MODELS
    if MODELS is None:
        MODELS = joblib.load(OUT / "models.joblib")
    return MODELS


def score(p: dict) -> dict:
    row = pd.DataFrame([{
        "Area": p["Area"], "VehPower": int(p["VehPower"]), "VehAge": int(p["VehAge"]),
        "DrivAge": int(p["DrivAge"]), "BonusMalus": int(p["BonusMalus"]), "VehBrand": p["VehBrand"],
        "VehGas": p["VehGas"], "Density": int(p["Density"]), "Region": p["Region"]}])
    m = models()
    f_glm = float(m["glm_f"].predict(glm_features(row).reindex(columns=m["glm_cols"], fill_value=0.0))[0])
    f_gbm = float(m["gbm_f"].predict(gbm_features(row))[0])
    s_glm = float(m["sev_glm"].predict(glm_features(row).reindex(columns=m["glm_cols"], fill_value=0.0))[0]) * m["k_glm"]
    s_gbm = float(m["sev_gbm"].predict(gbm_features(row))[0])
    sev = (s_gbm if m["use_gbm_sev"] else s_glm / m["k_glm"]) * m["k_gbm"]
    return {"freq_glm": f_glm, "freq_gbm": f_gbm, "severity": sev,
            "pure_premium_glm": f_glm * s_glm, "pure_premium_gbm": f_gbm * sev,
            "p_claim_year": 1 - float(np.exp(-f_gbm))}


TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json",
         ".md": "text/markdown; charset=utf-8", ".svg": "image/svg+xml"}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/results":
            f = OUT / "results.json"
            return self._send(200, f.read_bytes()) if f.exists() else self._send(404, '{"error":"run python -m src.pipeline first"}')
        if path == "/api/report":
            f = ROOT / "reports" / "report.md"
            return self._send(200, f.read_bytes(), TYPES[".md"]) if f.exists() else self._send(404, "no report")
        rel = "index.html" if path == "/" else path.lstrip("/")
        f = (UI / rel).resolve()
        if UI.resolve() in f.parents and f.is_file():
            return self._send(200, f.read_bytes(), TYPES.get(f.suffix, "application/octet-stream"))
        self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/score":
            return self._send(404, "{}")
        try:
            n = int(self.headers.get("Content-Length", 0))
            self._send(200, json.dumps(score(json.loads(self.rfile.read(n)))))
        except Exception as e:  # surface bad input to the UI
            self._send(400, json.dumps({"error": str(e)}))


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    if not (OUT / "models.joblib").exists():
        sys.exit("No trained models found. Run:  python -m src.pipeline")
    models()
    print(f"Claims model validation UI -> http://localhost:{port}  (Ctrl+C to stop)")
    try:
        webbrowser.open(f"http://localhost:{port}")
    except Exception:
        pass
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()

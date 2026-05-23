"""LitServe REST API for bfe-pm.

Launch:
    python api/server.py
    python api/server.py --checkpoint_dir /path/to/checkpoints --port 8080

Request (POST /predict):
    {
        "operation_text": "腹腔鏡膽囊切除術",
        "scheduled_duration": 90,
        "anesthesia": "GA",
        "surgery_category": "常規刀",
        "shift": "白班",
        "weekday": 1
    }

Response:
    {
        "predicted_time_min": 74.2,
        "interval_90": [52.6, 121.8],
        "interval_method": "split_conformal",
        "model_name": "bfe_pm_ensemble_v1"
    }
"""
import argparse
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import litserve as ls
from litserve import LitAPI, LitServer

from bfe_pm import BfePmPredictor
from bfe_pm.preprocess import WEEKDAY_ZH


class SurgeryDurationAPI(LitAPI):
    def setup(self, device):
        checkpoint_dir = getattr(self, "_checkpoint_dir", None)
        self.predictor = BfePmPredictor(
            device=device,
            local_checkpoint_dir=checkpoint_dir,
        )

    def decode_request(self, request):
        body = request
        weekday = body.get("weekday")
        if weekday is None:
            weekday = WEEKDAY_ZH.get(body.get("weekday_str", ""), 1)
        return {
            "operation_text":  str(body.get("operation_text", "")),
            "scheduled_duration": float(body.get("scheduled_duration", 0)),
            "anesthesia":      str(body.get("anesthesia",      "GA")),
            "surgery_category": str(body.get("surgery_category", "常規刀")),
            "shift":           str(body.get("shift",           "白班")),
            "weekday":         int(weekday),
            "is_daytime":      bool(body.get("is_daytime",     False)),
            "is_outpatient":   bool(body.get("is_outpatient",  False)),
        }

    def predict(self, inputs):
        t0 = time.perf_counter()
        result = self.predictor.predict(**inputs, conformal=True)
        result["inference_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        result["request_id"]   = uuid.uuid4().hex
        return result

    def encode_response(self, output):
        return output


class IOLogger(ls.Logger):
    def process(self, key, value):
        print(f"[log] {key} = {value}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--accelerator", default="cpu")
    args = parser.parse_args()

    api = SurgeryDurationAPI()
    api._checkpoint_dir = args.checkpoint_dir

    server = LitServer(
        api,
        accelerator=args.accelerator,
        loggers=[IOLogger()],
        model_metadata={"name": "bfe_pm_ensemble", "version": "1.0.0"},
    )
    server.run(port=args.port)

"""High-level inference API for bfe-pm ensemble."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from .conformal import cqr_interval, load_thresholds, split_conformal_interval
from .model import BERT_NAME, DualStreamBERT, DualStreamBERTQuantile
from .preprocess import WEEKDAY_ZH, build_struct_tensor, tokenize_operation

HF_REPO = "your-username/bfe-pm"
SEEDS = [0, 1, 42]
_LOCAL_CACHE = Path.home() / ".cache" / "bfe_pm"


def _load_checkpoint(seed: int, local_dir: Optional[Path] = None) -> Path:
    fname = f"bfe_pm_seed{seed}_best.pt"
    if local_dir and (local_dir / fname).exists():
        return local_dir / fname
    _LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
    cached = _LOCAL_CACHE / fname
    if cached.exists():
        return cached
    print(f"Downloading {fname} from HuggingFace Hub …")
    path = hf_hub_download(repo_id=HF_REPO, filename=fname, local_dir=str(_LOCAL_CACHE))
    return Path(path)


class BfePmPredictor:
    """3-seed ensemble predictor with optional conformal intervals.

    Usage::

        predictor = BfePmPredictor()
        result = predictor.predict(
            operation_text="腹腔鏡膽囊切除術",
            scheduled_duration=90,
            anesthesia="GA",
            surgery_category="常規刀",
            shift="白班",
            weekday=1,
        )
        print(result)
        # {'point_pred_min': 74.2, 'interval_90': [52.6, 121.8], 'method': 'split_conformal'}
    """

    def __init__(
        self,
        device: Optional[str] = None,
        local_checkpoint_dir: Optional[str] = None,
        bert_name: str = BERT_NAME,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        local_dir = Path(local_checkpoint_dir) if local_checkpoint_dir else None

        self.tokenizer = AutoTokenizer.from_pretrained(bert_name)
        self.models: List[DualStreamBERT] = []
        self.scaler_mean: float = 0.0
        self.scaler_scale: float = 1.0

        for seed in SEEDS:
            ckpt_path = _load_checkpoint(seed, local_dir)
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            if seed == SEEDS[0]:
                sc = ckpt["scaler_y"]
                self.scaler_mean  = float(sc.mean_[0])
                self.scaler_scale = float(sc.scale_[0])
            model = DualStreamBERT(bert_name=bert_name, wide_struct=True).to(self.device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            self.models.append(model)

        self._thresholds = load_thresholds()
        print(f"bfe-pm ensemble ready ({len(self.models)} seeds) on {self.device}")

    def _inverse_transform(self, val_scaled: float) -> float:
        return val_scaled * self.scaler_scale + self.scaler_mean

    def predict(
        self,
        operation_text: str,
        scheduled_duration: float = 0.0,
        anesthesia: str = "GA",
        surgery_category: str = "常規刀",
        shift: str = "白班",
        weekday: int = 1,
        is_daytime: bool = False,
        is_outpatient: bool = False,
        weekday_str: Optional[str] = None,
        conformal: bool = True,
    ) -> Dict:
        """Predict surgical duration.

        Args:
            operation_text: Chinese procedure name(s), joined with ' [SEP] '
            scheduled_duration: surgeon estimated minutes (0 = unknown)
            anesthesia: GA | EPI | SA | MAC | Local | Block | IV
            surgery_category: 常規刀 | 急診刀_Urgent | 急刀_Emergency | 日間手術 | 門診手術
            shift: 白班 | 小夜 | 大夜
            weekday: 0=Monday … 6=Sunday (overrides weekday_str)
            is_daytime: 日間手術 flag
            is_outpatient: 門診手術 flag
            weekday_str: 星期一 … 星期天 (convenience alias)
            conformal: if True, also return 90% split conformal interval

        Returns:
            dict with keys: point_pred_min, interval_90, interval_method, model_name
        """
        if weekday_str:
            weekday = WEEKDAY_ZH.get(weekday_str, weekday)

        input_ids, attention_mask = tokenize_operation(
            operation_text, self.tokenizer, device=self.device
        )
        struct = build_struct_tensor(
            scheduled_duration, anesthesia, surgery_category,
            shift, weekday, is_daytime, is_outpatient,
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds_scaled = [
                m(input_ids, attention_mask, struct).item() for m in self.models
            ]
        ensemble_scaled = sum(preds_scaled) / len(preds_scaled)
        point_pred = self._inverse_transform(ensemble_scaled)

        result: Dict = {
            "point_pred_min": round(point_pred, 1),
            "model_name": "bfe_pm_ensemble_v1",
        }

        if conformal:
            q_hat = self._thresholds["split_conformal"]["q_hat_min"]
            lo, hi = split_conformal_interval(point_pred, q_hat)
            result["interval_90"] = [round(lo, 1), round(hi, 1)]
            result["interval_method"] = "split_conformal"
            result["interval_coverage_target"] = 0.90

        return result

    def predict_batch(self, cases: List[Dict]) -> List[Dict]:
        """Predict a list of cases. Each dict follows predict() kwargs."""
        return [self.predict(**case) for case in cases]

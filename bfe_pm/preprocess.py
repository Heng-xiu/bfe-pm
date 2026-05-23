"""Feature encoding for inference — matches training dataset.py exactly."""
import math
from typing import Optional

import torch
from transformers import AutoTokenizer

ANESTHESIA_TYPES = ["EPI", "GA", "SA", "MAC", "Local", "Local(不照會麻醉科)", "Block", "IV", "other"]
SURGERY_CATS     = ["急診刀_Urgent", "常規刀", "急刀_Emergency", "日間手術", "門診手術", "other"]
SHIFT_TYPES      = ["白班", "小夜", "大夜", "unknown"]
WEEKDAY_ZH = {"星期天": 0, "星期一": 1, "星期二": 2, "星期三": 3,
               "星期四": 4, "星期五": 5, "星期六": 6}


def _safe_encode(val: str, categories: list) -> int:
    val = str(val).strip()
    for i, c in enumerate(categories):
        if val in c or c in val:
            return i
    return len(categories) - 1


def build_struct_tensor(
    scheduled_duration: float,
    anesthesia: str,
    surgery_category: str,
    shift: str,
    weekday: int,
    is_daytime: bool = False,
    is_outpatient: bool = False,
) -> torch.Tensor:
    """Encode 8 structured features into a float32 tensor.

    Args:
        scheduled_duration: surgeon-estimated minutes (0 if unknown)
        anesthesia: e.g. "GA", "EPI", "SA"
        surgery_category: e.g. "常規刀", "急診刀_Urgent"
        shift: "白班" | "小夜" | "大夜"
        weekday: 0=Monday … 6=Sunday
        is_daytime: 日間手術 flag
        is_outpatient: 門診手術 flag
    """
    feats = [
        min(float(scheduled_duration) / 300.0, 3.0),
        _safe_encode(anesthesia,       ANESTHESIA_TYPES) / len(ANESTHESIA_TYPES),
        _safe_encode(surgery_category, SURGERY_CATS)     / len(SURGERY_CATS),
        _safe_encode(shift,            SHIFT_TYPES)      / len(SHIFT_TYPES),
        math.sin(2 * math.pi * weekday / 7),
        math.cos(2 * math.pi * weekday / 7),
        1.0 if is_daytime   else 0.0,
        1.0 if is_outpatient else 0.0,
    ]
    return torch.tensor(feats, dtype=torch.float32)


def tokenize_operation(
    operation_text: str,
    tokenizer: AutoTokenizer,
    max_len: int = 128,
    device: Optional[torch.device] = None,
):
    """Tokenize procedure name for BERT input."""
    enc = tokenizer(
        operation_text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    if device is not None:
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)
    return enc["input_ids"], enc["attention_mask"]

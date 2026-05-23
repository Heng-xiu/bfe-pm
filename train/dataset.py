"""SurgeryDataset and data loading utilities — verbatim from training experiments."""
import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

ANESTHESIA_TYPES = ["EPI", "GA", "SA", "MAC", "Local", "Local(不照會麻醉科)", "Block", "IV", "other"]
SURGERY_CATS     = ["急診刀_Urgent", "常規刀", "急刀_Emergency", "日間手術", "門診手術", "other"]
SHIFT_TYPES      = ["白班", "小夜", "大夜", "unknown"]


def _safe_encode(val: str, categories: List[str]) -> int:
    val = str(val).strip()
    for i, c in enumerate(categories):
        if val in c or c in val:
            return i
    return len(categories) - 1


class SurgeryDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        scaler_y: Optional[StandardScaler],
        max_len: int = 128,
        max_samples: Optional[int] = None,
    ):
        df = df.reset_index(drop=True)
        if max_samples:
            df = df.iloc[:max_samples]
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

        y = df["duration_min"].values.reshape(-1, 1).astype(np.float32)
        if scaler_y is None:
            self.scaler_y = StandardScaler()
            self.y = self.scaler_y.fit_transform(y).flatten()
        else:
            self.scaler_y = scaler_y
            self.y = scaler_y.transform(y).flatten()

    def _struct(self, row) -> torch.Tensor:
        sched = float(row.get("預定耗時", 0) or 0)
        return torch.tensor([
            min(sched / 300.0, 3.0),
            _safe_encode(str(row.get("麻醉方式", "")), ANESTHESIA_TYPES) / len(ANESTHESIA_TYPES),
            _safe_encode(str(row.get("手術類別", "")), SURGERY_CATS)     / len(SURGERY_CATS),
            _safe_encode(str(row.get("班別",     "")), SHIFT_TYPES)      / len(SHIFT_TYPES),
            float(row.get("weekday_sin", 0)),
            float(row.get("weekday_cos", 0)),
            float(row.get("日間手術_bin", 0)),
            float(row.get("門診手術_bin", 0)),
        ], dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            str(row.get("operation_text", "") or ""),
            truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt",
        )
        return (
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            self._struct(row),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


def load_splits(
    parquet_path: str,
    tokenizer_name: str = "trueto/medbert-kd-chinese",
    max_len: int = 128,
    seed: int = 42,
    batch_size: int = 64,
    num_workers: int = 2,
    max_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    df = pd.read_parquet(parquet_path)
    strat = df["手術類別"].fillna("other").astype(str)
    vc = strat.value_counts()
    strat = strat.replace({k: "other" for k in vc[vc < 20].index})

    idx = np.arange(len(df))
    idx_train, idx_tmp = train_test_split(idx, test_size=0.30, random_state=seed, stratify=strat)
    strat_tmp = strat.iloc[idx_tmp].reset_index(drop=True)
    idx_val, idx_test = train_test_split(idx_tmp, test_size=0.50, random_state=seed, stratify=strat_tmp)

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    train_ds = SurgeryDataset(df.iloc[idx_train], tok, None,   max_len, max_samples)
    val_ds   = SurgeryDataset(df.iloc[idx_val],   tok, train_ds.scaler_y, max_len)
    test_ds  = SurgeryDataset(df.iloc[idx_test],  tok, train_ds.scaler_y, max_len)

    make = lambda ds, s: DataLoader(ds, batch_size=batch_size, shuffle=s,
                                    num_workers=num_workers, pin_memory=True)
    return make(train_ds, True), make(val_ds, False), make(test_ds, False), train_ds.scaler_y

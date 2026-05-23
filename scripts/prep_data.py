"""Preprocess surgical_records.xls → processed.parquet.

Usage:
    python scripts/prep_data.py --input surgical_records.xls --output data/processed.parquet
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def roc_to_western(s: str) -> str:
    return str(int(s[:-4]) + 1911) + s[-4:]


def derive_shift(dt_series):
    h = dt_series.dt.hour
    shift = pd.Series("白班", index=dt_series.index)
    shift[h >= 16] = "小夜"
    shift[h < 8]   = "大夜"
    return shift


def main(args):
    print(f"Loading {args.input} ...")
    df = pd.read_excel(args.input, engine="xlrd" if args.input.endswith(".xls") else "openpyxl")
    print(f"Raw rows: {len(df)}")

    for col in ["手術結束時間", "麻醉開始時間"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.dropna(subset=["手術結束時間", "麻醉開始時間"]).copy()
    df["duration_min"] = (df["手術結束時間"] - df["麻醉開始時間"]).dt.total_seconds() / 60.0
    df = df[(df["duration_min"] >= 5) & (df["duration_min"] <= 720)].copy()
    print(f"After filter: {len(df)} rows")

    df["手術日_dt"] = pd.to_datetime(
        df["手術日"].astype(str).str.strip().apply(
            lambda s: roc_to_western(s) if len(s) == 7 else s
        ), format="%Y%m%d", errors="coerce",
    )
    df["weekday"]     = df["手術日_dt"].dt.dayofweek
    df["weekday_sin"] = np.sin(2 * math.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * math.pi * df["weekday"] / 7)

    if "班別" not in df.columns and "入手術室時間" in df.columns:
        df["入手術室時間"] = pd.to_datetime(df["入手術室時間"], errors="coerce")
        df["班別"] = derive_shift(df["入手術室時間"]).fillna("白班")
    elif "班別" not in df.columns:
        df["班別"] = "unknown"

    df["預定耗時"] = df["預定耗時"].fillna(df["預定耗時"].median())
    df["日間手術_bin"] = (df["日間手術"].astype(str).str.strip().str.upper() == "Y").astype(int)
    df["門診手術_bin"] = (df["門診手術"].astype(str).str.strip().str.upper() == "Y").astype(int)

    op_cols = [c for c in ["預定術式1名稱","預定術式2名稱","預定術式3名稱","預定術式4名稱"] if c in df.columns]
    df["operation_text"] = (
        df[op_cols].fillna("").apply(
            lambda r: " [SEP] ".join(v.strip() for v in r if v.strip()), axis=1
        )
    )
    op_code_cols = [c for c in ["預定術式1","預定術式2","預定術式3","預定術式4"] if c in df.columns]
    df["op_code_1"] = df[op_code_cols[0]].fillna("__none__").astype(str) if op_code_cols else "__none__"

    keep = ["SCHEDULE_ID","duration_min","預定耗時","麻醉方式","手術類別","班別",
            "日間手術_bin","門診手術_bin","weekday","weekday_sin","weekday_cos",
            "operation_text","op_code_1"]
    df = df[[c for c in keep if c in df.columns]]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Saved {len(df)} rows → {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--output", default="data/processed.parquet")
    main(p.parse_args())

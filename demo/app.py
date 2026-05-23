"""bfe-pm Gradio Demo — Surgical Duration Prediction with Conformal Intervals.

Launch:
    python demo/app.py
    python demo/app.py --checkpoint_dir /path/to/checkpoints
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent))
from bfe_pm import BfePmPredictor

# ── Pre-computed calibration statistics (from paper, held-out calibration split test set) ────────
PAPER_STATS = {
    "mae_min": 33.8,
    "coverage_90": 89.6,
    "interval_width_min": 153.9,
    "duration_percentiles": {
        "p10": 25, "p25": 50, "p50": 90, "p75": 155, "p90": 265
    },
}

ANESTHESIA_OPTS   = ["GA", "EPI", "SA", "MAC", "Local", "Local(不照會麻醉科)", "Block", "IV"]
SURGERY_CAT_OPTS  = ["常規刀", "急診刀_Urgent", "急刀_Emergency", "日間手術", "門診手術"]
SHIFT_OPTS        = ["白班", "小夜", "大夜"]
WEEKDAY_OPTS      = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期天"]
WEEKDAY_MAP       = {d: i for i, d in enumerate(["星期一","星期二","星期三","星期四","星期五","星期六","星期天"])}

predictor: BfePmPredictor = None


def load_model(checkpoint_dir: str = None):
    global predictor
    if predictor is None:
        predictor = BfePmPredictor(local_checkpoint_dir=checkpoint_dir)


def predict_and_plot(
    operation_text: str,
    scheduled_duration: float,
    anesthesia: str,
    surgery_category: str,
    shift: str,
    weekday_str: str,
    is_daytime: bool,
    is_outpatient: bool,
):
    if not operation_text.strip():
        return "⚠️ 請輸入術式名稱", None, None

    result = predictor.predict(
        operation_text=operation_text.strip(),
        scheduled_duration=scheduled_duration,
        anesthesia=anesthesia,
        surgery_category=surgery_category,
        shift=shift,
        weekday=WEEKDAY_MAP.get(weekday_str, 0),
        is_daytime=is_daytime,
        is_outpatient=is_outpatient,
        conformal=True,
    )

    pred  = result["point_pred_min"]
    lo    = result["interval_90"][0]
    hi    = result["interval_90"][1]

    # ── Summary text ─────────────────────────────────────────────────────────
    urgency_note = ""
    if "急" in surgery_category:
        urgency_note = (
            "\n\n⚠️ **注意**：急診手術類別的 conformal 覆蓋率可能低於 90%（參見論文 §6）。"
            "此信賴區間適用於擇期手術（常規刀）。"
        )

    summary = f"""## 預測結果

| 項目 | 數值 |
|------|------|
| **預測手術時長** | **{pred:.1f} 分鐘** |
| **90% 信賴區間** | [{lo:.1f}, {hi:.1f}] 分鐘 |
| 區間寬度 | {hi - lo:.1f} 分鐘 |
| 模型 | bfe-pm 3-seed ensemble |
| 信賴區間方法 | Split Conformal (q̂ = 76.95 min) |

> 90% 信賴區間代表：在相似的擇期手術案例中，約 90% 的實際手術時長落在此區間內。{urgency_note}
"""

    # ── Figure 1: Timeline bar ────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(8, 2.5))
    ax1.set_xlim(0, max(hi * 1.15, scheduled_duration * 1.2, 30))
    ax1.set_ylim(0, 1)

    # Conformal interval
    ax1.barh(0.5, hi - lo, left=lo, height=0.35,
             color="#4C9BE8", alpha=0.35, label=f"90% 信賴區間 [{lo:.0f}–{hi:.0f} min]")
    # Point prediction
    ax1.axvline(pred, color="#1a6fbd", linewidth=2.5, label=f"預測值 {pred:.1f} min")
    # Scheduled duration
    if scheduled_duration > 0:
        ax1.axvline(scheduled_duration, color="#E05C5C", linewidth=2,
                    linestyle="--", label=f"預訂時長 {scheduled_duration:.0f} min")

    ax1.set_xlabel("手術時長（分鐘）", fontsize=11)
    ax1.set_yticks([])
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_title("預測時長 vs 90% 信賴區間", fontsize=12, fontweight="bold")
    ax1.spines[["top", "right", "left"]].set_visible(False)
    plt.tight_layout()

    # ── Figure 2: Population distribution ────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    p = PAPER_STATS["duration_percentiles"]
    # Approximate log-normal distribution from percentiles
    mu   = np.log(p["p50"])
    sigma = (np.log(p["p90"]) - np.log(p["p10"])) / (2 * 1.282)
    xs = np.linspace(0, 500, 400)
    ys = np.exp(-(np.log(np.maximum(xs, 1)) - mu) ** 2 / (2 * sigma ** 2)) / (xs * sigma * np.sqrt(2 * np.pi) + 1e-6)
    ys[xs <= 0] = 0
    ys = ys / ys.max()

    ax2.fill_between(xs, ys, alpha=0.20, color="#4C9BE8", label="手術時長分佈（估計）")
    ax2.plot(xs, ys, color="#4C9BE8", linewidth=1.5)

    # Shade interval
    mask = (xs >= lo) & (xs <= hi)
    ax2.fill_between(xs[mask], ys[mask], alpha=0.45, color="#1a6fbd", label="90% 信賴區間")
    ax2.axvline(pred, color="#1a6fbd", linewidth=2.5, label=f"預測值 {pred:.0f} min")

    ax2.set_xlabel("手術時長（分鐘）", fontsize=11)
    ax2.set_ylabel("相對密度", fontsize=11)
    ax2.set_title("預測值於資料集分佈中的位置", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 450)
    ax2.set_ylim(0, None)
    ax2.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    return summary, fig1, fig2


def build_ui(checkpoint_dir: str = None):
    load_model(checkpoint_dir)

    with gr.Blocks(title="bfe-pm 手術時長預測", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
# 🏥 bfe-pm — 手術時長預測 Demo

基於 **MedBERT-KD-Chinese + 寬結構特徵 MLP** 的雙流架構，提供點預測與 90% Conformal 信賴區間。

> **論文**：*Conformalized Uncertainty Quantification for Surgical Duration Prediction* (IEEE JBHI, submitted)
> **MAE**：33.8 分鐘（擇期手術） · **90% 覆蓋率**：89.6%
        """)

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 手術資訊輸入")
                op_text = gr.Textbox(
                    label="預定術式名稱（中文）",
                    placeholder="例：腹腔鏡膽囊切除術 [SEP] 腹腔鏡探查術",
                    lines=2,
                )
                sched_dur = gr.Number(label="預訂耗時（分鐘，不確定可填 0）", value=0, minimum=0)

                with gr.Row():
                    anesthesia = gr.Dropdown(
                        ANESTHESIA_OPTS, label="麻醉方式", value="GA"
                    )
                    surgery_cat = gr.Dropdown(
                        SURGERY_CAT_OPTS, label="手術類別", value="常規刀"
                    )

                with gr.Row():
                    shift   = gr.Dropdown(SHIFT_OPTS,   label="班別",   value="白班")
                    weekday = gr.Dropdown(WEEKDAY_OPTS, label="手術星期", value="星期二")

                with gr.Row():
                    is_daytime   = gr.Checkbox(label="日間手術", value=False)
                    is_outpatient = gr.Checkbox(label="門診手術", value=False)

                predict_btn = gr.Button("▶ 預測", variant="primary", size="lg")

            with gr.Column(scale=3):
                result_md = gr.Markdown("*填入術式資訊後點擊「預測」*")

        with gr.Row():
            timeline_plot = gr.Plot(label="預測時長 vs 信賴區間")
            dist_plot     = gr.Plot(label="於資料集分佈中的位置")

        predict_btn.click(
            fn=predict_and_plot,
            inputs=[op_text, sched_dur, anesthesia, surgery_cat, shift, weekday,
                    is_daytime, is_outpatient],
            outputs=[result_md, timeline_plot, dist_plot],
        )

        gr.Markdown("""
---
### 範例術式

| 術式 | 建議麻醉 | 預估時長 |
|------|---------|---------|
| 腹腔鏡膽囊切除術 | GA | 60–90 min |
| 剖腹產 | SA / EPI | 45–75 min |
| 全髖關節置換術 | SA | 90–150 min |
| 冠狀動脈繞道手術 | GA | 180–360 min |

### 注意事項
- 90% 信賴區間使用 **Split Conformal Prediction**（q̂ = 76.95 min），在擇期手術（常規刀）達到 89.6% 實際覆蓋率
- 急診手術因殘差分佈偏移，覆蓋率可能低於目標值（詳見論文 §6）
- 本 demo 僅供研究參考，不作為臨床決策依據
        """)

        gr.Examples(
            examples=[
                ["腹腔鏡膽囊切除術", 60, "GA", "常規刀", "白班", "星期二", False, False],
                ["剖腹產手術 [SEP] 子宮下段橫切剖腹取胎術", 60, "SA", "常規刀", "白班", "星期三", False, False],
                ["全髖關節置換術", 120, "SA", "常規刀", "白班", "星期四", False, False],
                ["冠狀動脈繞道手術 [SEP] 體外循環下心臟手術", 240, "GA", "常規刀", "白班", "星期一", False, False],
                ["急診闌尾切除術", 60, "GA", "急診刀_Urgent", "小夜", "星期五", False, False],
            ],
            inputs=[op_text, sched_dur, anesthesia, surgery_cat, shift, weekday,
                    is_daytime, is_outpatient],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Local path to bfe_pm_seed*.pt files")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_ui(args.checkpoint_dir)
    demo.launch(server_port=args.port, share=args.share)

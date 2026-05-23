"""Generate bfe-pm end-to-end flow diagram → assets/diagram.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ── palette ──────────────────────────────────────────────────────────────────
C_BG      = "#F8F9FB"
C_TEXT    = "#1B2A4A"
C_BERT    = "#2563EB"      # blue  — text stream
C_STRUCT  = "#059669"      # green — struct stream
C_FUSION  = "#7C3AED"      # purple — fusion / ensemble
C_CONF    = "#D97706"      # amber — conformal
C_OUTPUT  = "#DC2626"      # red   — output
C_ARROW   = "#64748B"
C_BORDER  = "#E2E8F0"

plt.rcParams['font.sans-serif'] = [
    'PingFang HK', 'STHeiti', 'Arial Unicode MS',
    'Noto Sans CJK JP', 'Noto Sans CJK SC', 'Droid Sans Fallback',
    'DejaVu Sans',
]
plt.rcParams['axes.unicode_minus'] = False

FIG_W, FIG_H = 16, 10
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)


# ── helpers ───────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, color, alpha=1.0, radius=0.25, lw=0, edgecolor=None):
    ec = edgecolor or color
    p = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=color, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=3,
    )
    ax.add_patch(p)
    return p

def label(ax, x, y, text, color="white", size=10, weight="bold", ha="center", va="center", zorder=4):
    ax.text(x, y, text, color=color, fontsize=size, fontweight=weight,
            ha=ha, va=va, zorder=zorder, linespacing=1.4)

def arrow(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.5, style="->"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=2)

def dim_tag(ax, x, y, text, color):
    ax.text(x, y, text, color=color, fontsize=8, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=color, linewidth=1.2), zorder=5)


# ════════════════════════════════════════════════════════════════════════════════
# TITLE
# ════════════════════════════════════════════════════════════════════════════════
ax.text(FIG_W/2, 9.5, "bfe-pm  End-to-End Inference Flow",
        color=C_TEXT, fontsize=18, fontweight="bold", ha="center", va="center", zorder=4)
ax.text(FIG_W/2, 9.1,
        "MedBERT-KD-Chinese + Wide Struct MLP  ·  3-Seed Ensemble  ·  Conformal Prediction",
        color="#64748B", fontsize=10, ha="center", va="center", zorder=4)

# ════════════════════════════════════════════════════════════════════════════════
# ROW 1 — INPUTS
# ════════════════════════════════════════════════════════════════════════════════
# Text input box
box(ax, 4.2, 8.1, 5.0, 1.1, C_BERT, alpha=0.12, lw=1.8, edgecolor=C_BERT)
label(ax, 4.2, 8.35, "術式名稱（中文文字）", color=C_BERT, size=10)
label(ax, 4.2, 7.95, "腹腔鏡膽囊切除術 [SEP] 腹腔鏡探查術",
      color=C_TEXT, size=8.5, weight="normal")

# Struct input box
box(ax, 11.5, 8.1, 5.8, 1.1, C_STRUCT, alpha=0.12, lw=1.8, edgecolor=C_STRUCT)
label(ax, 11.5, 8.35, "結構化特徵（8 維）", color=C_STRUCT, size=10)
label(ax, 11.5, 7.95,
      "麻醉方式  ·  手術類別  ·  班別  ·  星期（sin/cos）  ·  預定耗時",
      color=C_TEXT, size=8.5, weight="normal")

# Input section label
ax.text(0.35, 8.1, "INPUT", color="#94A3B8", fontsize=9, fontweight="bold",
        ha="center", va="center", rotation=90)

# ════════════════════════════════════════════════════════════════════════════════
# ARROWS input → encoder
# ════════════════════════════════════════════════════════════════════════════════
arrow(ax, 4.2, 7.55, 4.2, 6.75, color=C_BERT)
arrow(ax, 11.5, 7.55, 11.5, 6.75, color=C_STRUCT)

# ════════════════════════════════════════════════════════════════════════════════
# ROW 2 — ENCODERS
# ════════════════════════════════════════════════════════════════════════════════
# BERT box
box(ax, 4.2, 6.2, 5.0, 1.0, C_BERT, alpha=0.9, radius=0.3)
label(ax, 4.2, 6.42, "MedBERT-KD-Chinese", color="white", size=10.5)
label(ax, 4.2, 6.05, "12 層  ·  凍結 0–7 層  ·  CLS → Linear(768→256) → GELU",
      color="#DBEAFE", size=8, weight="normal")

# Struct MLP box
box(ax, 11.5, 6.2, 5.8, 1.0, C_STRUCT, alpha=0.9, radius=0.3)
label(ax, 11.5, 6.42, "Wide Struct MLP", color="white", size=10.5)
label(ax, 11.5, 6.05, "8 → 192 → LayerNorm → Dropout → 64  ·  GELU",
      color="#D1FAE5", size=8, weight="normal")

# Encoder section label
ax.text(0.35, 6.2, "ENCODE", color="#94A3B8", fontsize=9, fontweight="bold",
        ha="center", va="center", rotation=90)

# dim tags
dim_tag(ax, 4.2, 5.6, "256 dims", C_BERT)
dim_tag(ax, 11.5, 5.6, "64 dims", C_STRUCT)

# ════════════════════════════════════════════════════════════════════════════════
# ARROWS encoder → concat
# ════════════════════════════════════════════════════════════════════════════════
arrow(ax, 4.2, 5.45, 4.2, 5.05, color=C_BERT)
arrow(ax, 11.5, 5.45, 11.5, 5.05, color=C_STRUCT)
# converge lines to fusion
ax.plot([4.2, 7.85], [4.95, 4.95], color=C_ARROW, lw=1.5, zorder=2)
ax.plot([11.5, 7.85], [4.95, 4.95], color=C_ARROW, lw=1.5, zorder=2)
arrow(ax, 7.85, 4.95, 7.85, 4.55, color=C_ARROW)
dim_tag(ax, 7.85, 4.78, "concat → 320 dims", C_FUSION)

# ════════════════════════════════════════════════════════════════════════════════
# ROW 3 — FUSION MLP
# ════════════════════════════════════════════════════════════════════════════════
box(ax, 7.85, 4.05, 4.2, 0.85, C_FUSION, alpha=0.9, radius=0.3)
label(ax, 7.85, 4.27, "Fusion MLP", color="white", size=10.5)
label(ax, 7.85, 3.92, "Linear(320→128) → GELU → Dropout → Linear(128→1)",
      color="#EDE9FE", size=8, weight="normal")

ax.text(0.35, 4.05, "FUSE", color="#94A3B8", fontsize=9, fontweight="bold",
        ha="center", va="center", rotation=90)

# ════════════════════════════════════════════════════════════════════════════════
# ARROWS fusion → 3 seeds
# ════════════════════════════════════════════════════════════════════════════════
arrow(ax, 7.85, 3.62, 7.85, 3.25, color=C_FUSION)

# seed fan-out
SEEDS_X = [4.5, 7.85, 11.2]
SEEDS_Y = 2.85
for sx in SEEDS_X:
    ax.plot([7.85, sx], [3.15, SEEDS_Y + 0.35], color=C_FUSION, lw=1.3,
            linestyle="--", zorder=2)

# ════════════════════════════════════════════════════════════════════════════════
# ROW 4 — SEEDS
# ════════════════════════════════════════════════════════════════════════════════
for sx, seed in zip(SEEDS_X, ["Seed 0", "Seed 1", "Seed 42"]):
    box(ax, sx, SEEDS_Y, 2.2, 0.65, C_FUSION, alpha=0.2, lw=1.5, edgecolor=C_FUSION)
    label(ax, sx, SEEDS_Y + 0.12, seed, color=C_FUSION, size=9.5)
    label(ax, sx, SEEDS_Y - 0.18, "pred = f(x; theta_seed)", color=C_TEXT, size=8, weight="normal")

ax.text(0.35, SEEDS_Y, "ENSEMBLE", color="#94A3B8", fontsize=9, fontweight="bold",
        ha="center", va="center", rotation=90)

# converge seeds → ensemble mean
for sx in SEEDS_X:
    ax.plot([sx, 7.85], [SEEDS_Y - 0.33, 2.08], color=C_FUSION, lw=1.3,
            linestyle="--", zorder=2)
arrow(ax, 7.85, 2.08, 7.85, 1.72, color=C_FUSION)
dim_tag(ax, 7.85, 1.9, "ensemble mean (3 seeds)", C_FUSION)

# ════════════════════════════════════════════════════════════════════════════════
# ROW 5 — CONFORMAL METHODS
# ════════════════════════════════════════════════════════════════════════════════
CONF_Y = 1.2
CONF_X = [3.1, 7.85, 12.6]

# fan out
for cx in CONF_X:
    ax.plot([7.85, cx], [1.62, CONF_Y + 0.38], color=C_CONF, lw=1.3,
            linestyle=":", zorder=2)

conf_data = [
    ("Split Conformal",  "±76.95 min\nWidth: 153.9 min"),
    ("CQR Ensemble",     "Q10/Q90 + 13.21 min\nWidth: 126.8 min  ✦ narrowest"),
    ("Mondrian",         "Quintile-stratified\nWidth: 143.8 min"),
]
for (cx, (title, sub)) in zip(CONF_X, conf_data):
    box(ax, cx, CONF_Y, 3.6, 0.72, C_CONF, alpha=0.15, lw=1.5, edgecolor=C_CONF)
    label(ax, cx, CONF_Y + 0.18, title, color=C_CONF, size=9.5)
    label(ax, cx, CONF_Y - 0.17, sub.replace("✦", "*"), color=C_TEXT, size=7.8, weight="normal")

ax.text(0.35, CONF_Y, "CONFORMAL", color="#94A3B8", fontsize=8, fontweight="bold",
        ha="center", va="center", rotation=90)

# ════════════════════════════════════════════════════════════════════════════════
# COVERAGE badge
# ════════════════════════════════════════════════════════════════════════════════
box(ax, 7.85, 0.42, 5.5, 0.6, C_OUTPUT, alpha=0.9, radius=0.28)
label(ax, 7.85, 0.58, "Point Prediction  +  90% Conformal Interval", color="white", size=10.5)
label(ax, 7.85, 0.27, "pred = 74.2 min   |   PI-90 = [31.2, 118.2] min   |   MAE 33.8 min   |   Coverage 90.1%",
      color="#FEE2E2", size=8.5, weight="normal")

arrow(ax, 7.85, 0.84, 7.85, 0.72, color=C_OUTPUT)
ax.text(0.35, 0.42, "OUTPUT", color="#94A3B8", fontsize=9, fontweight="bold",
        ha="center", va="center", rotation=90)

# ════════════════════════════════════════════════════════════════════════════════
# vertical section dividers
# ════════════════════════════════════════════════════════════════════════════════
for y in [7.55, 5.5, 3.45, 2.3, 0.82]:
    ax.axhline(y, color=C_BORDER, lw=0.8, zorder=1, xmin=0.04, xmax=0.98)

# ════════════════════════════════════════════════════════════════════════════════
# footer
# ════════════════════════════════════════════════════════════════════════════════
ax.text(FIG_W/2, 0.06,
        "github.com/Heng-xiu/bfe-pm  ·  huggingface.co/Heng666/bfe-pm  ·  MIT License",
        color="#94A3B8", fontsize=8, ha="center", va="center")

# ════════════════════════════════════════════════════════════════════════════════
# save
# ════════════════════════════════════════════════════════════════════════════════
import os
out = os.path.join(os.path.dirname(__file__), "..", "assets", "diagram.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
print(f"Saved → {os.path.abspath(out)}")

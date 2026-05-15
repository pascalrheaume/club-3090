"""Generate 3090Ti power-cap efficiency chart from @pascalrheaume's air-cooled rig.

Source data: 2026-05-14 sweep, single 3090Ti rig (GPU 0 used), air-cooled.
Engine: mainline llama.cpp (ghcr.io/ggml-org/llama.cpp:server-cuda) +
Qwen3.6-27B-UD-Q3_K_XL.gguf, decode-single (10s × 2 timed streams)

Sweep methodology: time-bounded streaming bench (10s/direction at each cap).
Total wall: ~10m 23caps from 230-450W in 10W increments. The time-bounded
approach (vs token-bounded) makes per-cap wall constant ~23s regardless of cap,
so total runtime scales linearly with cap count, not throttle severity.

Sampling fields: GPU temp, SM clock, memory clock, power-throttle %, P-state per cap
(median of in-load samples where util>50%). The extended sweep reveals the full
efficiency curve up to 450W, with peak efficiency at 310W (0.107 TPS/W) and
a gradual decline beyond 330W as the GPU draws more power without proportional
gains in throughput.
"""
import matplotlib.pyplot as plt

# (cap_W, narr_TPS, code_TPS, actual_W, gpu_temp_C, sm_clk_MHz, mem_clk_MHz, pwr_throttle_pct, p_state, eff_TPS_per_W) — 23-cap extended sweep
data = [
    (230, 19.88, 20.28, 229.69, 63, 750, 10251, 100, "P2", 0.087),
    (240, 22.07, 21.87, 239.54, 64, 825, 10251, 97, "P2", 0.092),
    (250, 23.57, 23.17, 249.40, 65, 885, 10251, 100, "P2", 0.095),
    (260, 25.17, 25.37, 259.04, 65, 975, 10251, 100, "P2", 0.097),
    (270, 26.77, 26.67, 269.08, 66, 1035, 10251, 100, "P2", 0.099),
    (280, 28.27, 28.57, 279.07, 67, 1125, 10251, 100, "P2", 0.101),
    (290, 30.27, 30.27, 289.19, 67, 1215, 10251, 100, "P2", 0.105),
    (300, 31.77, 31.76, 299.83, 68, 1290, 10251, 100, "P2", 0.106),
    (310, 33.46, 33.36, 309.38, 69, 1365, 10251, 100, "P2", 0.108),
    (320, 34.76, 34.86, 319.43, 69, 1470, 10251, 100, "P2", 0.109),
    (330, 36.36, 36.26, 329.20, 70, 1635, 10251, 100, "P2", 0.110), 
    (340, 37.36, 37.46, 339.45, 70, 1725, 10251, 100, "P2", 0.110),
    (350, 38.45, 37.96, 348.79, 71, 1755, 10251, 100, "P2", 0.110), #⭐ sweet spot
    (360, 38.96, 38.86, 358.66, 72, 1770, 10251, 97, "P2", 0.109),
    (370, 39.25, 39.16, 368.89, 72, 1815, 10251, 97, "P2", 0.106),
    (380, 39.46, 39.26, 378.75, 73, 1845, 10251, 100, "P2", 0.104),
    (390, 39.86, 39.76, 388.42, 73, 1860, 10251, 100, "P2", 0.103),
    (400, 40.06, 39.66, 398.67, 74, 1875, 10251, 100, "P2", 0.100),
    (410, 40.06, 40.16, 407.98, 75, 1890, 10251, 97, "P2", 0.098),
    (420, 40.35, 39.96, 418.79, 75, 1905, 10251, 97, "P2", 0.096),
    (430, 40.45, 40.36, 428.56, 76, 1920, 10251, 97, "P2", 0.094),
    (440, 40.55, 40.56, 438.52, 76, 1935, 10251, 100, "P2", 0.092),
    (450, 40.45, 40.76, 448.35, 77, 1935, 10251, 100, "P2", 0.090), # stock TDP
]

caps = [d[0] for d in data]
narr = [d[1] for d in data]
code = [d[2] for d in data]
draw = [d[3] for d in data]
gpu_temp = [d[4] for d in data]
sm_clk = [d[5] for d in data]
mem_clk = [d[6] for d in data]
pwr_throttle = [d[7] for d in data]
p_state = [d[8] for d in data]
eff = [d[9] for d in data]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

fig, ax1 = plt.subplots(figsize=(11, 6.4), dpi=150)

# Left axis: TPS
color_narr = "#1f77b4"
color_code = "#2ca02c"
ax1.plot(caps, narr, "o-", color=color_narr, linewidth=2.2, markersize=6,
         label="Narrative TPS", zorder=3)
ax1.plot(caps, code, "s-", color=color_code, linewidth=2.2, markersize=6,
         label="Code TPS", zorder=3)
ax1.set_xlabel("Power cap (W)", fontsize=13)
ax1.set_ylabel("Wall TPS (single-stream, llama.cpp mainline)", fontsize=13)
ax1.set_xlim(225, 455)
ax1.set_ylim(14, 43)
ax1.grid(True, alpha=0.3, zorder=0)
ax1.tick_params(axis="both", labelsize=11)

# Right axis: TPS/W efficiency
ax2 = ax1.twinx()
color_eff = "#d62728"
ax2.plot(caps, eff, "^--", color=color_eff, linewidth=1.8, markersize=5,
         alpha=0.9, label="Efficiency (narr TPS/W)", zorder=2)
ax2.set_ylabel("Efficiency: TPS/W (narrative)", color=color_eff, fontsize=13)
ax2.tick_params(axis="y", labelcolor=color_eff, labelsize=11)
ax2.set_ylim(0.07, 0.118)

# Sweet spot annotation: 350W
ax1.axvline(350,color="goldenrod", linestyle=":", alpha=0.5, linewidth=1.5)
ax1.annotate(
    "★ 350W cap\n0.11 TPS/W (best efficiency)\n38.45 narr / 37.98 code\nSM 1755 MHz, 78% of stock TDP",
    xy=(350, 38),
    xytext=(320, 28),
    fontsize=10.5,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3cd", edgecolor="goldenrod", linewidth=1.2),
    arrowprops=dict(arrowstyle="->", color="goldenrod", lw=1.5),
    zorder=4,
)

# Stock TDP marker at 450W
ax1.axvline(450, color="#888", linestyle="--", alpha=0.6, linewidth=1.2)
ax1.annotate(
    "stock TDP\n450W (GPU 0)",
    xy=(450, 40.06),
    xytext=(455, 42),
    fontsize=10,
    ha="left",
    color="#555",
    fontstyle="italic",
)

# Title
ax1.set_title(
    "RTX 3090Ti Qwen3.6-27B + llama.cpp — extended power-cap efficiency curve",
    pad=14,
)

# Subtitle
fig.text(
    0.5, 0.92,
    "1× 3090Ti air-cooled (GPU 0), mainline llama.cpp + Q3_K_XL GGUF, "
    "time-bounded single-stream, 230-450W sweep  |  data: @pascalrheaume (2026-05-14)",
    ha="center", fontsize=10, color="#666",
    style="italic",
)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc="lower right", fontsize=11, framealpha=0.95,
           edgecolor="#ccc")

# Footer
fig.text(
    0.99, 0.01,
    "github.com/pascalrheaume/club-3090",
    ha="right", fontsize=9, color="#888", style="italic",
)

plt.tight_layout(rect=(0, 0.02, 1, 0.92))

out = "/tmp/power_cap_sweep_3090Ti_Qwen36.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")

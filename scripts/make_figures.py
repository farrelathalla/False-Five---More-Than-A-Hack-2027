"""Draw the README and technical-document figures from the files in this repository.

Inputs: results/*.csv, results/report.json, submission/False_Five.csv and
artifacts/final_probabilities.npy. No match data is needed.

Usage:
    python scripts/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

H_C, A_C, D_C, X_C = "#2B4FA3", "#D7263D", "#BCC3CC", "#FFFFFF"
ACC, INK, MUTED = "#0F9D8A", "#1F2933", "#7B8794"
SRC_C = {"dfl": "#4C6EF5", "metrica": "#0F9D8A", "pff": "#F59F00"}
SRC_N = {"dfl": "DFL (Bundesliga)", "metrica": "Metrica sample", "pff": "PFF World Cup 2022"}
LAB_C = {"H": H_C, "A": A_C, "D": D_C, "X": X_C}
FPS = 25

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#CBD2D9",
    "axes.labelcolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": INK,
    "ytick.color": INK, "axes.grid": True, "grid.color": "#EEF1F4", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "legend.frameon": False,
})

report = json.load(open(RES / "report.json"))
sub = pd.read_csv(ROOT / "submission" / "False_Five.csv")
labels = sub.label.values
prob = np.load(ROOT / "artifacts" / "final_probabilities.npy")
cfg = json.load(open(ROOT / "artifacts" / "decoder.json"))


def runs(lab, s, e):
    out, i = [], s
    while i < e:
        j = i
        while j < e and lab[j] == lab[i]:
            j += 1
        out.append((i, j, lab[i]))
        i = j
    return out


def strip(ax, segs, t0=0, scale=1.0):
    for s, e, l in segs:
        ax.axvspan((s - t0) * scale, (e - t0) * scale, color=LAB_C[l], lw=0)
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def save(fig, name):
    fig.savefig(FIG / name, facecolor="white")
    plt.close(fig)
    print("wrote", FIG / name)


# 1. public window: truth, prediction and probabilities
gt = pd.read_csv(RES / "public_gt_segments.csv")
a, b = 11185, 18685
fig, ax = plt.subplots(3, 1, figsize=(12, 4.6), sharex=True,
                       gridspec_kw=dict(height_ratios=[0.8, 0.8, 2.2], hspace=0.25))
sec = 1 / FPS
strip(ax[0], [(r.start, r.end, r.label) for r in gt.itertuples()], a, sec)
strip(ax[1], runs(labels, a, b), a, sec)
ax[0].set_ylabel("ground\ntruth", rotation=0, ha="right", va="center")
ax[1].set_ylabel("False Five", rotation=0, ha="right", va="center")
t = (np.arange(a, b) - a) * sec
ax[2].plot(t, prob[a:b, 0], color=H_C, lw=1.1, label="P(Barcelona)")
ax[2].plot(t, prob[a:b, 1], color=A_C, lw=1.1, label="P(Athletic)")
ax[2].plot(t, prob[a:b, 2], color="#52606D", lw=1.0, label="P(dead ball)")
ax[2].set_ylim(-0.03, 1.03)
ax[2].set_ylabel("ensemble\nprobability")
ax[2].set_xlabel("seconds into the public window (frames 11185 to 18684)")
ax[2].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32))
ax[2].set_xlim(0, (b - a) * sec)
fig.legend(handles=[Patch(color=H_C, label="H  Barcelona"), Patch(color=A_C, label="A  Athletic"),
                    Patch(color=D_C, label="D  dead ball")], ncol=3, loc="upper center",
           bbox_to_anchor=(0.5, 1.02))
save(fig, "public_window.png")

# 2. full match timeline and possession durations
fig = plt.figure(figsize=(12, 4.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[2.2, 2.2, 1.6], hspace=0.6, wspace=0.35)
for k, (s, e) in enumerate(cfg["halves"]):
    axh = fig.add_subplot(gs[k, :2])
    strip(axh, runs(labels, s, e), s, 1 / (FPS * 60))
    axh.set_xlim(0, (e - s) / (FPS * 60))
    axh.set_ylabel(f"half {k + 1}", rotation=0, ha="right", va="center")
    axh.spines["bottom"].set_visible(True)
    axh.set_xlabel("minutes from the start of the graded half" if k == 1 else "")
axd = fig.add_subplot(gs[:, 2])
durs = {"H": [], "A": []}
for s, e in cfg["halves"]:
    for rs, re_, l in runs(labels, s, e):
        if l in durs:
            durs[l].append((re_ - rs) / FPS)
bins = np.arange(0, 92, 4)
axd.hist([np.clip(durs["H"], 0, 90), np.clip(durs["A"], 0, 90)], bins=bins, stacked=True,
         color=[H_C, A_C], label=["Barcelona", "Athletic"])
axd.set_xlabel("possession length (s)")
axd.set_ylabel("possessions")
axd.set_title(f"{report['full_match']['possessions']} possessions, mean "
              f"{report['full_match']['mean_possession_s']:.1f} s", fontsize=10)
axd.legend()
fig.suptitle("Full-match prediction: possessions and stoppages over both halves", fontweight="bold", y=1.02)
save(fig, "full_match.png")

# 3. held-out spellF1 per model and source
mo = pd.read_csv(RES / "model_oof.csv", index_col=0)
mo.loc["ensemble"] = [report["oof"].get(c, np.nan) for c in mo.columns]
names = {"gbm": "LightGBM", "gru_A": "BiGRU (A)", "gru_B": "BiGRU (B)", "tcn_A": "TCN (A)",
         "tcn_B": "TCN (B)", "ensemble": "Ensemble +\ndecoder"}
fig, ax = plt.subplots(figsize=(11, 3.9))
x = np.arange(len(mo))
w = 0.2
for j, src in enumerate(["dfl", "metrica", "pff"]):
    ax.bar(x + (j - 1.5) * w, mo[f"{src}_spellF1"], w, color=SRC_C[src], label=SRC_N[src])
ax.bar(x + 1.5 * w, mo["spellF1"], w, color=INK, label="weighted mean")
for i, v in enumerate(mo["spellF1"]):
    ax.text(i + 1.5 * w, v + 0.006, f"{v:.3f}", ha="center", fontsize=8.5, color=INK, fontweight="bold")
ax.set_xticks(x, [names[n] for n in mo.index])
ax.set_ylim(0.55, 0.86)
ax.set_ylabel("held-out spellF1")
ax.set_title("Out-of-fold spellF1 on 52 held-out halves from three leagues and three trackers")
ax.legend(ncol=4, loc="upper left", fontsize=9)
ax.axvspan(len(mo) - 1.5, len(mo) - 0.5, color="#E6FCF5", zorder=0)
save(fig, "model_scores.png")

# 4. greedy ensemble selection
es = pd.read_csv(RES / "ensemble_selection.csv")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw=dict(width_ratios=[1.7, 1]))
path = es[es.step > 0]
ax[0].plot(path.step, path.spellF1, "-o", color=ACC, lw=2, ms=6)
for r in path.itertuples():
    ax[0].annotate(names[r.added].replace("\n", " "), (r.step, r.spellF1), textcoords="offset points",
                   xytext=(0, 8), ha="center", fontsize=7.5, color=MUTED, rotation=30)
ref = es[es.step == 0].spellF1.iloc[0]
ax[0].axhline(ref, ls="--", color=MUTED, lw=1)
ax[0].text(10, ref - 0.0015, "plain 50/50 blend (LightGBM + BiGRU)", ha="right", va="top", fontsize=8, color=MUTED)
ax[0].set_xlabel("selection step (one member copy added per step)")
ax[0].set_ylabel("OOF spellF1, reference decoder")
ax[0].set_ylim(ref - 0.006, path.spellF1.max() + 0.008)
ax[0].set_title("Greedy forward selection with replacement")
wts = {k: v for k, v in report["ensemble_weights"].items() if v > 0}
cols = {"gru_A": "#364FC7", "gru_B": "#748FFC", "tcn_A": "#0B7285", "tcn_B": "#66D9E8"}
ax[1].pie(list(wts.values()), labels=[f"{names[k]}\n{v:.0%}" for k, v in wts.items()],
          colors=[cols[k] for k in wts], startangle=90, counterclock=False,
          wedgeprops=dict(width=0.42, edgecolor="white"), textprops=dict(fontsize=9))
ax[1].set_title("Final ensemble weights")
save(fig, "ensemble.png")

# 5. decoder grid
dg = pd.read_csv(RES / "decoder_grid.csv")
g1 = dg[dg.min_dead == 1].pivot_table(index="c_dead", columns="c_team", values="spellF1")
fig, ax = plt.subplots(figsize=(7.4, 3.6))
im = ax.imshow(g1.values, cmap="viridis", aspect="auto", origin="lower")
ax.set_xticks(range(len(g1.columns)), [f"{c:g}" for c in g1.columns])
ax.set_yticks(range(len(g1.index)), [f"{c:g}" for c in g1.index])
ax.set_xlabel("c_team  (cost of a direct H/A switch)")
ax.set_ylabel("c_dead  (cost to enter\nor leave a stoppage)")
ax.grid(False)
for i in range(g1.shape[0]):
    for j in range(g1.shape[1]):
        v = g1.values[i, j]
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7.5,
                color="white" if v < g1.values.mean() else INK)
ci, cj = list(g1.index).index(cfg["c_dead"]), list(g1.columns).index(cfg["c_team"])
ax.add_patch(plt.Rectangle((cj - 0.5, ci - 0.5), 1, 1, fill=False, ec="#FF6B6B", lw=2.5))
fig.colorbar(im, ax=ax, label="OOF spellF1")
ax.set_title("Viterbi switch costs chosen on held-out spellF1")
save(fig, "decoder_grid.png")

# 6. label alignment
la = pd.read_csv(RES / "label_alignment.csv")
bar = la[la.source == "BAR public GT"].iloc[0]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for src in ["dfl", "metrica", "pff"]:
    d = la[la.source == src].copy()
    d["tau"] = d.tau.astype(int)
    ax[0].plot(d.tau / FPS, d.spells_per_alive_min, "-o", color=SRC_C[src], label=SRC_N[src], lw=2)
    ax[1].plot(d.tau / FPS, d.median_s, "-o", color=SRC_C[src], label=SRC_N[src], lw=2)
for k, (col, lab) in enumerate([("spells_per_alive_min", "possessions per live minute"), ("median_s", "median possession (s)")]):
    ax[k].axhline(float(bar[col]), color=INK, ls="--", lw=1.2)
    ax[k].text(-0.1, float(bar[col]), "BAR-ATH public truth ", va="bottom", ha="left", fontsize=8.5, color=INK)
    ax[k].set_xlabel("tau: shortest spell that counts as control (s)")
    ax[k].set_ylabel(lab)
ax[0].legend(fontsize=8.5)
ax[0].set_title("Aligning every source to controlled possession")
ax[1].set_title("Possession length after alignment")
save(fig, "label_alignment.png")

# 7. missing-ball evidence
lr = report["decoder"]["gap_logratios"]
cats = ["ball\nvisible", "gap\n<0.5 s", "0.5 to\n2 s", "2 to\n5 s", "5 to\n10 s", "10 to\n30 s", "gap\n>30 s"]
fig, ax = plt.subplots(figsize=(8, 3.3))
ax.bar(cats, lr, color=[H_C if v < 0 else "#52606D" for v in lr])
for i, v in enumerate(lr):
    ax.text(i, v + (0.1 if v >= 0 else -0.35), f"{v:+.2f}", ha="center", fontsize=9)
ax.axhline(0, color=INK, lw=0.8)
ax.set_ylabel("log P(category | dead)\n- log P(category | live)")
ax.set_title("Missing-ball evidence estimated on the match itself (one EM step)")
save(fig, "gap_evidence.png")

# 8. public scoreboard vs official baselines
board = [("False Five", 0.6875, 0.9267), ("baseline smooth125", 0.4242, 0.5812),
         ("baseline smooth50", 0.3256, 0.6207), ("baseline smooth25", 0.2868, 0.6200),
         ("baseline raw", 0.0378, 0.5972)]
fig, ax = plt.subplots(1, 2, figsize=(11, 2.9), sharey=True)
yy = np.arange(len(board))[::-1]
for k, (title, idx) in enumerate([("spellF1 (ranking metric)", 1), ("macroF1 (tie-break)", 2)]):
    vals = [r[idx] for r in board]
    ax[k].barh(yy, vals, color=[ACC] + ["#CBD2D9"] * 4)
    for y, v in zip(yy, vals):
        ax[k].text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=9)
    ax[k].set_xlim(0, 1.08)
    ax[k].set_title(title)
ax[0].set_yticks(yy, [r[0] for r in board])
fig.suptitle("Official scorer on the public 5-minute window", fontweight="bold", y=1.04)
save(fig, "scoreboard.png")

# 9. per-half OOF distribution
ph = pd.read_csv(RES / "oof_per_half.csv")
fig, ax = plt.subplots(figsize=(8, 3.3))
rng = np.random.default_rng(0)
for i, src in enumerate(["dfl", "metrica", "pff"]):
    v = ph[ph.source == src].spellF1.values
    ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, color=SRC_C[src], alpha=0.75, s=28)
    ax.hlines(np.mean(v), i - 0.3, i + 0.3, color=INK, lw=2.2)
    ax.text(i + 0.33, np.mean(v), f"{np.mean(v):.3f}", va="center", fontsize=9)
ax.set_xticks(range(3), [f"{SRC_N[s]}\n({(ph.source == s).sum()} halves)" for s in ["dfl", "metrica", "pff"]])
ax.set_ylabel("spellF1 per held-out half")
ax.set_title("Every held-out half, every source")
save(fig, "oof_per_half.png")

# 10. boundary timing
bo = pd.read_csv(RES / "boundary_offsets.csv")
bo = bo[bo.source != "BAR public"]
kinds = {"team": "team change", "enter_D": "ball goes dead", "leave_D": "restart"}
fig, ax = plt.subplots(figsize=(8, 3.2))
for j, (kind, lab) in enumerate(kinds.items()):
    for i, src in enumerate(["dfl", "metrica", "pff"]):
        r = bo[(bo.source == src) & (bo.kind == kind)].iloc[0]
        y = j * 4 + i
        ax.plot([r.q25 / FPS, r.q75 / FPS], [y, y], color=SRC_C[src], lw=5, alpha=0.45, solid_capstyle="round")
        ax.plot(r["median"] / FPS, y, "o", color=SRC_C[src], ms=7)
ax.axvline(0, color=INK, lw=1)
ax.set_yticks([j * 4 + 1 for j in range(3)], list(kinds.values()))
ax.set_xlabel("predicted minus true boundary (s): median (dot) and interquartile range (bar)")
ax.legend(handles=[Patch(color=SRC_C[s], label=SRC_N[s]) for s in SRC_C], ncol=3, fontsize=8.5,
          loc="upper center", bbox_to_anchor=(0.5, 1.2))
ax.invert_yaxis()
save(fig, "boundary_timing.png")

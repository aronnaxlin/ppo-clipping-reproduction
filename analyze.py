"""Aggregate runs into the paper's Table 1 metric, per-env tables, and figures."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_all import CLIPS, ENVS, SEEDS

VARIANTS = ["noclip" if c == "none" else f"clip{float(c)}" for c in CLIPS]
LABELS = {"noclip": "No clipping", "clip0.1": "Clip ε=0.1", "clip0.2": "Clip ε=0.2", "clip0.3": "Clip ε=0.3"}
PAPER_TABLE1 = {"noclip": -0.39, "clip0.1": 0.76, "clip0.2": 0.82, "clip0.3": 0.70}
# categorical slots 1-4 in fixed order (validated: CVD ΔE 9.1, contrast relief via legend + tables)
COLORS = {"noclip": "#2a78d6", "clip0.1": "#eb6834", "clip0.2": "#1baf7a", "clip0.3": "#eda100"}
STYLES = {"noclip": "--", "clip0.1": "-", "clip0.2": "-", "clip0.3": "-"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
GRID_STEP, WINDOW = 10_000, 20_000


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def load_runs(root):
    rows = []
    for env in ENVS:
        for v in VARIANTS:
            for seed in SEEDS:
                d = Path(root) / env / v / f"seed{seed}"
                if (d / "done.json").exists():
                    rows.append({**json.loads((d / "done.json").read_text()), "dir": d})
    return pd.DataFrame(rows)


def curve(run_dir, total):
    ep = pd.read_csv(run_dir / "episodes.csv")
    grid = np.arange(GRID_STEP, total + 1, GRID_STEP)
    out = np.full(len(grid), np.nan)
    for i, g in enumerate(grid):
        m = ep[(ep.global_step > g - WINDOW) & (ep.global_step <= g)]["return"]
        if len(m):
            out[i] = m.mean()
    return grid, pd.Series(out).ffill().to_numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    df = load_runs(args.runs)
    rand = json.loads(Path("random_baseline.json").read_text())
    expected = len(ENVS) * len(VARIANTS) * len(SEEDS)
    print(f"{len(df)}/{expected} runs found")

    # ---- paper's normalized score: random = 0, best run in that env = 1 ----
    best = df.groupby("env_id")["last100_mean_return"].max()
    df["norm"] = [(r.last100_mean_return - rand[r.env_id]) / (best[r.env_id] - rand[r.env_id])
                  for r in df.itertuples()]

    rng = np.random.default_rng(0)
    table1 = []
    for v in VARIANTS:
        sub = df[df.variant == v]
        if sub.empty:
            continue
        # stratified bootstrap: resample seeds within each env, then average over all runs
        boots = []
        groups = [g["norm"].to_numpy() for _, g in sub.groupby("env_id")]
        for _ in range(10_000):
            boots.append(np.mean(np.concatenate([rng.choice(g, len(g)) for g in groups])))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        table1.append({"variant": v, "setting": LABELS[v], "runs": len(sub), "ours": sub["norm"].mean(),
                       "ci95_low": lo, "ci95_high": hi, "paper": PAPER_TABLE1[v]})
    t1 = pd.DataFrame(table1)
    t1["rank_ours"] = t1["ours"].rank(ascending=False).astype(int)
    t1["rank_paper"] = t1["paper"].rank(ascending=False).astype(int)

    per_env = (df.groupby(["env_id", "variant"])["last100_mean_return"]
               .agg(["mean", "std", "count"]).reset_index())
    norm_env = df.pivot_table(index="env_id", columns="variant", values="norm", aggfunc="mean")[VARIANTS]

    with open(out / "tables.md", "w") as f:
        f.write("## Table 1 reproduction: average normalized score\n\n")
        f.write("| Setting | Runs | Ours | 95% CI (stratified bootstrap) | Rank | Paper | Paper rank |\n")
        f.write("| :--- | ---: | ---: | :---: | ---: | ---: | ---: |\n")
        for r in t1.itertuples():
            f.write(f"| {r.setting} | {r.runs} | {r.ours:.2f} | [{r.ci95_low:.2f}, {r.ci95_high:.2f}] | "
                    f"{r.rank_ours} | {r.paper:.2f} | {r.rank_paper} |\n")
        f.write("\n## Normalized score per environment (mean over seeds)\n\n| Env | "
                + " | ".join(LABELS[v] for v in VARIANTS) + " |\n| :--- |" + " ---: |" * len(VARIANTS) + "\n")
        for env, row in norm_env.iterrows():
            f.write(f"| {env} | " + " | ".join(f"{row[v]:.2f}" for v in VARIANTS) + " |\n")
        f.write("\n## Raw return, last 100 training episodes (mean ± std over seeds)\n\n| Env | Random | "
                + " | ".join(LABELS[v] for v in VARIANTS) + " |\n| :--- | ---: |" + " ---: |" * len(VARIANTS) + "\n")
        for env in ENVS:
            cells = []
            for v in VARIANTS:
                m = per_env[(per_env.env_id == env) & (per_env.variant == v)]
                cells.append(f"{m['mean'].iloc[0]:.1f} ± {m['std'].iloc[0]:.1f}" if len(m) else "—")
            f.write(f"| {env} | {rand[env]:.1f} | " + " | ".join(cells) + " |\n")
    df.drop(columns="dir").to_csv(out / "runs.csv", index=False)

    # ---- learning curves on a common step grid, one row per (run, grid point) ----
    curve_rows = []
    for r in df.itertuples():
        grid, ret = curve(r.dir, 1_000_000)
        curve_rows += [(r.env_id, r.variant, r.seed, int(g), v) for g, v in zip(grid, ret)]
    pd.DataFrame(curve_rows, columns=["env_id", "variant", "seed", "global_step", "return_20k_window"]) \
        .to_csv(out / "curves.csv", index=False, float_format="%.4g")
    print((out / "tables.md").read_text())

    # ---- learning curves: small multiples, mean ± std over seeds ----
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4), facecolor=SURFACE)
    for ax, env in zip(axes.flat, ENVS):
        style_axes(ax)
        for v in VARIANTS:
            dirs = df[(df.env_id == env) & (df.variant == v)]["dir"]
            if dirs.empty:
                continue
            curves = np.array([curve(d, 1_000_000)[1] for d in dirs])
            grid = np.arange(GRID_STEP, 1_000_001, GRID_STEP)
            m, s = np.nanmean(curves, 0), np.nanstd(curves, 0)
            ax.fill_between(grid, m - s, m + s, color=COLORS[v], alpha=0.15, linewidth=0)
            ax.plot(grid, m, color=COLORS[v], linestyle=STYLES[v], linewidth=2, label=LABELS[v])
        ax.axhline(rand[env], color=INK2, linewidth=1, linestyle=":")
        ax.set_title(env, fontsize=10, color=INK)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:g}M"))
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK2, linewidth=1, linestyle=":"))
    labels.append("Random policy")
    legend_ax.legend(handles, labels, loc="center", frameon=False, fontsize=10, labelcolor=INK)
    fig.supxlabel("Timesteps", color=INK2, fontsize=10)
    fig.supylabel("Episode return (training, 20k-step window)", color=INK2, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=200)

    # ---- diagnostic: how far each objective lets the policy move per update ----
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4), facecolor=SURFACE)
    for ax, env in zip(axes.flat, ENVS):
        style_axes(ax)
        for v in VARIANTS:
            dirs = df[(df.env_id == env) & (df.variant == v)]["dir"]
            if dirs.empty:
                continue
            ups = [pd.read_csv(d / "updates.csv") for d in dirs]
            kl = np.median(np.array([u["approx_kl"].to_numpy() for u in ups]), 0)
            ax.plot(ups[0]["global_step"], kl, color=COLORS[v], linestyle=STYLES[v], linewidth=2, label=LABELS[v])
        ax.set_yscale("log")
        ax.set_title(env, fontsize=10, color=INK)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:g}M"))
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    legend_ax.legend(*axes.flat[0].get_legend_handles_labels(), loc="center", frameon=False, fontsize=10,
                     labelcolor=INK)
    fig.supxlabel("Timesteps", color=INK2, fontsize=10)
    fig.supylabel("Approx. KL(old ‖ new) per update, median over seeds", color=INK2, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "kl_per_update.png", dpi=200)

    # ---- headline: ours vs paper ----
    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=SURFACE)
    style_axes(ax)
    x = np.arange(len(t1))
    ax.bar(x - 0.2, t1["paper"], width=0.38, color="#b7b6b0", label="Paper (Gym -v1)")
    ax.bar(x + 0.2, t1["ours"], width=0.38, color="#2a78d6", label="Ours (Gymnasium -v5)")
    ax.errorbar(x + 0.2, t1["ours"], yerr=[t1["ours"] - t1["ci95_low"], t1["ci95_high"] - t1["ours"]],
                fmt="none", ecolor=INK, elinewidth=1, capsize=3)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks(x, t1["setting"], color=INK)
    ax.set_ylabel("Avg. normalized score", color=INK2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "table1_vs_paper.png", dpi=200)
    print(f"figures written to {out}/")


if __name__ == "__main__":
    main()

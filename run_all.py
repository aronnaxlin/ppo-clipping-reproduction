"""Run the full clipping ablation grid in parallel; finished runs (done.json) are skipped."""

import argparse
import itertools
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ENVS = ["HalfCheetah-v5", "Hopper-v5", "InvertedDoublePendulum-v5", "InvertedPendulum-v5",
        "Reacher-v5", "Swimmer-v5", "Walker2d-v5"]  # paper footnote 2, now -v5
CLIPS = ["none", "0.1", "0.2", "0.3"]
SEEDS = [1, 2, 3]


def run_dir(out, env, clip, seed):
    variant = "noclip" if clip == "none" else f"clip{float(clip)}"
    return Path(out) / env / variant / f"seed{seed}"


def launch(out, env, clip, seed):
    d = run_dir(out, env, clip, seed)
    d.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with open(d / "stdout.log", "w") as log:
        rc = subprocess.call([sys.executable, "ppo.py", "--env-id", env, "--clip", clip,
                              "--seed", str(seed), "--out-dir", out], stdout=log, stderr=subprocess.STDOUT)
    return env, clip, seed, rc, time.time() - t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--workers", type=int, default=20)
    args = p.parse_args()

    jobs = [j for j in itertools.product(ENVS, CLIPS, SEEDS)
            if not (run_dir(args.out_dir, *j) / "done.json").exists()]
    print(f"{len(jobs)} runs to go, {args.workers} workers", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(launch, args.out_dir, *j) for j in jobs]
        for n, f in enumerate(as_completed(futures), 1):
            env, clip, seed, rc, dt = f.result()
            status = "ok" if rc == 0 else f"FAILED rc={rc}"
            print(f"[{n}/{len(jobs)}] {env} clip={clip} seed={seed} {status} {dt / 60:.1f}min", flush=True)


if __name__ == "__main__":
    main()

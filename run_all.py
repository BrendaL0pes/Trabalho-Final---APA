"""Driver: run solve_one.py over a dataset (or a slice of it), with a
per-instance time budget derived from instance size, writing solutions to
solution/output/<dataset>/instance_XXXX.txt and appending a log line to
solution/run_log.tsv.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def budget_for(num_orders):
    if num_orders <= 200:
        return 30.0  # exact MILP path
    if num_orders < 2000:
        return 20.0
    if num_orders < 10000:
        return 35.0
    if num_orders < 30000:
        return 55.0
    return 80.0


def num_orders_of(path):
    with open(path) as f:
        return int(f.readline().split()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=20)
    args = ap.parse_args()

    log_path = os.path.join(HERE, "run_log.tsv")
    for n in range(args.start, args.end + 1):
        fname = f"instance_{n:04d}.txt"
        in_path = os.path.join(ROOT, "datasets", args.dataset, fname)
        if not os.path.exists(in_path):
            continue
        out_dir = os.path.join(HERE, "output", args.dataset)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, fname)

        o = num_orders_of(in_path)
        budget = budget_for(o)
        hard_timeout = int(budget + 60)

        t0 = time.time()
        try:
            res = subprocess.run(
                [sys.executable, os.path.join(HERE, "solve_one.py"), in_path, out_path,
                 "--budget", str(budget)],
                cwd=HERE, capture_output=True, text=True, timeout=hard_timeout,
            )
            line = res.stdout.strip() or res.stderr.strip()
        except subprocess.TimeoutExpired:
            line = f"{in_path}\tTIMEOUT after {hard_timeout}s"
        elapsed = time.time() - t0
        print(f"[{args.dataset} {n:04d}] budget={budget:.0f}s elapsed={elapsed:.1f}s :: {line}")
        with open(log_path, "a") as lf:
            lf.write(f"{args.dataset}\t{fname}\t{line}\telapsed={elapsed:.1f}\n")


if __name__ == "__main__":
    main()

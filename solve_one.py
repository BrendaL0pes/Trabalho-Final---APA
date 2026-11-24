import sys
import time
import argparse
from wop import Instance, write_solution, true_ratio
import heuristic
import exact


def choose_method(inst, force=None):
    if force:
        return force
    if inst.O <= 200:
        return "exact"
    return "heuristic"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_file")
    ap.add_argument("output_file")
    ap.add_argument("--budget", type=float, default=20.0)
    ap.add_argument("--method", choices=["exact", "heuristic"], default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    inst = Instance(args.input_file)
    method = choose_method(inst, args.method)

    if method == "exact":
        try:
            orders, aisles, ratio = exact.solve_exact(inst, args.budget)
        except Exception as e:
            print(f"exact solver failed ({e}); falling back to heuristic", file=sys.stderr)
            orders, aisles, ratio = heuristic.solve_heuristic(inst, args.budget, seed=args.seed)
            method = "heuristic(fallback)"
        if not aisles:
            orders, aisles, ratio = heuristic.solve_heuristic(inst, max(5.0, args.budget / 2), seed=args.seed)
            method = "heuristic(fallback-empty)"
    else:
        orders, aisles, ratio = heuristic.solve_heuristic(inst, args.budget, seed=args.seed)

    write_solution(args.output_file, orders, aisles)
    elapsed = time.time() - t0
    print(f"{args.input_file}\tO={inst.O}\tA={inst.A}\tmethod={method}\tratio={ratio:.6f}\t"
          f"orders={len(orders)}\taisles={len(aisles)}\ttime={elapsed:.1f}s")


if __name__ == "__main__":
    main()

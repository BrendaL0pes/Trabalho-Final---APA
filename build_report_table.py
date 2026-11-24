import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wop import Instance  # noqa: E402
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

spec = importlib.util.spec_from_file_location("checker", os.path.join(ROOT, "checker.py"))
checker_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker_mod)

best = {}
with open(os.path.join(ROOT, "best_solutions", "best_objectives.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        best[(row["dataset"], row["instance"])] = float(row["best_objective"])

rows = []
for dataset in ["a", "b", "x"]:
    files = sorted(os.listdir(os.path.join(ROOT, "datasets", dataset)))
    for fname in files:
        in_path = os.path.join(ROOT, "datasets", dataset, fname)
        out_path = os.path.join(HERE, "output", dataset, fname)
        if not os.path.exists(out_path):
            rows.append({"dataset": dataset, "instance": fname, "feasible": False,
                         "objective": 0.0, "best": best.get((dataset, fname)), "gap_pct": None,
                         "note": "no output file"})
            continue
        wop = checker_mod.WaveOrderPicking()
        wop.read_input(in_path)
        try:
            sel_o, sel_a = wop.read_output(out_path)
        except Exception as e:
            rows.append({"dataset": dataset, "instance": fname, "feasible": False,
                         "objective": 0.0, "best": best.get((dataset, fname)), "gap_pct": None,
                         "note": f"parse error: {e}"})
            continue
        feas = wop.is_solution_feasible(sel_o, sel_a)
        obj = wop.compute_objective_function(sel_o, sel_a) if feas else 0.0
        b = best.get((dataset, fname))
        gap = None
        if feas and b:
            gap = 100.0 * (b - obj) / b if b != 0 else 0.0
        rows.append({"dataset": dataset, "instance": fname, "feasible": feas,
                      "objective": obj, "best": b, "gap_pct": gap, "note": ""})

out_csv = os.path.join(HERE, "results_table.csv")
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dataset", "instance", "feasible", "objective", "best", "gap_pct", "note"])
    w.writeheader()
    for r in rows:
        w.writerow(r)

n_feas = sum(1 for r in rows if r["feasible"])
gaps = [r["gap_pct"] for r in rows if r["gap_pct"] is not None]
print(f"Total instances: {len(rows)}, feasible: {n_feas}")
if gaps:
    print(f"Average gap: {sum(gaps)/len(gaps):.2f}%  min={min(gaps):.2f}%  max={max(gaps):.2f}%")
for r in rows:
    print(r)

"""
Wave Order Picking Problem (SBPO 2025 / Mercado Livre challenge) solver.

Model
-----
Given O orders, I items, A aisles:
  - order o requires d[o][i] units of item i
  - aisle a stocks s[a][i] units of item i
  - choose X subset of orders, Y subset of aisles to maximize

        sum_{o in X} units(o)
        ----------------------         (1)
             |Y|

    subject to:
        LB <= sum_{o in X} units(o) <= UB                         (wave size)
        for every item i: sum_{o in X} d[o][i] <= sum_{a in Y} s[a][i]   (coverage)

This is a 0-1 fractional program. We solve it with Dinkelbach's parametric
algorithm: for a fixed q, the inner problem

        maximize  sum_{o in X} units(o) - q * |Y|                 (2)

is solved either EXACTLY (CP-SAT MILP, small instances) or HEURISTICALLY
(GRASP construction + local search, large instances). Dinkelbach updates
q <- units(X*)/|Y*| until it stops improving. The best *true* ratio found
at any iteration is kept (the heuristic inner solver is not exact, so the
classical monotonic-convergence guarantee is only approximate in practice).
"""
import sys
import time
import random
import numpy as np
from scipy import sparse


class Instance:
    def __init__(self, path):
        with open(path, "r") as f:
            lines = f.readlines()
        o, i, a = map(int, lines[0].split())
        self.O, self.I, self.A = o, i, a

        order_rows, order_cols, order_vals = [], [], []
        self.order_items = []  # list of (idx_array, val_array)
        for j in range(o):
            parts = lines[j + 1].split()
            d = int(parts[0])
            idx = np.empty(d, dtype=np.int64)
            val = np.empty(d, dtype=np.int64)
            for k in range(d):
                idx[k] = int(parts[2 * k + 1])
                val[k] = int(parts[2 * k + 2])
            self.order_items.append((idx, val))
            order_rows.extend([j] * d)
            order_cols.extend(idx.tolist())
            order_vals.extend(val.tolist())

        aisle_rows, aisle_cols, aisle_vals = [], [], []
        self.aisle_items = []
        for j in range(a):
            parts = lines[j + o + 1].split()
            d = int(parts[0])
            idx = np.empty(d, dtype=np.int64)
            val = np.empty(d, dtype=np.int64)
            for k in range(d):
                idx[k] = int(parts[2 * k + 1])
                val[k] = int(parts[2 * k + 2])
            self.aisle_items.append((idx, val))
            aisle_rows.extend([j] * d)
            aisle_cols.extend(idx.tolist())
            aisle_vals.extend(val.tolist())

        bounds = lines[o + a + 1].split()
        self.wave_lb = int(bounds[0])
        self.wave_ub = int(bounds[1])

        self.order_mat = sparse.csr_matrix(
            (order_vals, (order_rows, order_cols)), shape=(o, i), dtype=np.int64
        )
        self.aisle_mat = sparse.csr_matrix(
            (aisle_vals, (aisle_rows, aisle_cols)), shape=(a, i), dtype=np.int64
        )
        self.order_units = np.asarray(self.order_mat.sum(axis=1)).flatten()
        self.total_supply = np.asarray(self.aisle_mat.sum(axis=0)).flatten()

        # item -> list of (aisle_idx, qty) that stock it (for set-cover moves)
        self.item_to_aisles = [[] for _ in range(i)]
        for a_idx, (idx, val) in enumerate(self.aisle_items):
            for it, v in zip(idx.tolist(), val.tolist()):
                self.item_to_aisles[it].append((a_idx, v))


def write_solution(path, selected_orders, selected_aisles):
    so = sorted(selected_orders)
    sa = sorted(selected_aisles)
    with open(path, "w") as f:
        f.write(f"{len(so)}\n")
        for o in so:
            f.write(f"{o}\n")
        f.write(f"{len(sa)}\n")
        for a in sa:
            f.write(f"{a}\n")


def true_ratio(inst, selected_orders, selected_aisles):
    if not selected_aisles:
        return 0.0
    units = sum(inst.order_units[o] for o in selected_orders)
    return units / len(selected_aisles)

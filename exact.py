"""
Exact Dinkelbach + MILP inner solver (OR-Tools linear_solver / SCIP backend),
used only for small instances where the integer program is tractable within
the per-instance time budget.
"""
import time
from ortools.linear_solver import pywraplp


def solve_inner_milp(inst, q, time_limit_s, hint=None):
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        solver = pywraplp.Solver.CreateSolver("CBC")
    solver.SetTimeLimit(int(time_limit_s * 1000))

    x = [solver.BoolVar(f"x{o}") for o in range(inst.O)]
    y = [solver.BoolVar(f"y{a}") for a in range(inst.A)]

    total_units = solver.Sum([int(inst.order_units[o]) * x[o] for o in range(inst.O)])
    solver.Add(total_units >= inst.wave_lb)
    solver.Add(total_units <= inst.wave_ub)

    # coverage constraints, only for items that appear in at least one order
    I = inst.I
    order_mat = inst.order_mat.tocsc()
    aisle_mat = inst.aisle_mat.tocsc()
    for i in range(I):
        o_col = order_mat.getcol(i)
        if o_col.nnz == 0:
            continue
        a_col = aisle_mat.getcol(i)
        lhs = solver.Sum([int(o_col[r, 0]) * x[r] for r in o_col.nonzero()[0]])
        rhs = solver.Sum([int(a_col[r, 0]) * y[r] for r in a_col.nonzero()[0]])
        solver.Add(lhs <= rhs)

    objective = solver.Sum([int(inst.order_units[o]) * x[o] for o in range(inst.O)])
    objective -= q * solver.Sum(y)
    solver.Maximize(objective)

    if hint is not None:
        hx, hy = hint
        vars_, vals_ = [], []
        for o in range(inst.O):
            vars_.append(x[o])
            vals_.append(1 if o in hx else 0)
        for a in range(inst.A):
            vars_.append(y[a])
            vals_.append(1 if a in hy else 0)
        solver.SetHint(vars_, vals_)

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return None
    sel_o = {o for o in range(inst.O) if x[o].solution_value() > 0.5}
    sel_a = {a for a in range(inst.A) if y[a].solution_value() > 0.5}
    return sel_o, sel_a, status == pywraplp.Solver.OPTIMAL


def solve_exact(inst, time_budget, iter_time=None, max_iters=20):
    deadline = time.time() + time_budget
    q = 0.0
    best = (0.0, set(), set())
    hint = None
    it_budget = iter_time or max(2.0, time_budget / max_iters)
    for _ in range(max_iters):
        remaining = deadline - time.time()
        if remaining <= 0.5:
            break
        tl = min(it_budget, remaining)
        res = solve_inner_milp(inst, q, tl, hint=hint)
        if res is None:
            break
        sel_o, sel_a, optimal = res
        hint = (sel_o, sel_a)
        units = sum(int(inst.order_units[o]) for o in sel_o)
        if not sel_a or units < inst.wave_lb or units > inst.wave_ub:
            break
        ratio = units / len(sel_a)
        if ratio > best[0]:
            best = (ratio, sel_o, sel_a)
        # A timed-out inner solve is feasible, but it is not a valid Dinkelbach step.
        if not optimal:
            break
        if abs(ratio - q) < 1e-7:
            break
        q = ratio
    return sorted(best[1]), sorted(best[2]), best[0]

"""
GRASP construction + local search inner solver for the Dinkelbach outer loop,
used on instances too large for an exact MILP within the time budget.
"""
import random
import time
import numpy as np


class State:
    """Incremental (demand, supply) bookkeeping for a candidate (X, Y)."""

    def __init__(self, inst):
        self.inst = inst
        self.demand = np.zeros(inst.I, dtype=np.int64)
        self.supply = np.zeros(inst.I, dtype=np.int64)
        self.sel_orders = set()
        self.sel_aisles = set()
        self.total_units = 0

    def clone(self):
        s = State(self.inst)
        s.demand = self.demand.copy()
        s.supply = self.supply.copy()
        s.sel_orders = set(self.sel_orders)
        s.sel_aisles = set(self.sel_aisles)
        s.total_units = self.total_units
        return s

    def add_order_raw(self, o):
        idx, val = self.inst.order_items[o]
        self.demand[idx] += val
        self.sel_orders.add(o)
        self.total_units += int(self.inst.order_units[o])

    def remove_order_raw(self, o):
        idx, val = self.inst.order_items[o]
        self.demand[idx] -= val
        self.sel_orders.discard(o)
        self.total_units -= int(self.inst.order_units[o])

    def add_aisle_raw(self, a):
        idx, val = self.inst.aisle_items[a]
        self.supply[idx] += val
        self.sel_aisles.add(a)

    def remove_aisle_raw(self, a):
        idx, val = self.inst.aisle_items[a]
        self.supply[idx] -= val
        self.sel_aisles.discard(a)

    def shortfall_for_order(self, o):
        """Items (idx) and amounts by which supply would fall short if o is added now."""
        idx, val = self.inst.order_items[o]
        new_demand = self.demand[idx] + val
        slack = self.supply[idx] - new_demand
        short_mask = slack < 0
        if not short_mask.any():
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        return idx[short_mask], (-slack[short_mask])

    def cover_shortfall_greedy(self, short_idx, short_amt):
        """Greedily pick minimum-ish extra aisles (not yet selected) to cover a shortfall.
        Returns list of new aisle indices added (mutates self.supply/sel_aisles)."""
        if len(short_idx) == 0:
            return []
        remaining = dict(zip(short_idx.tolist(), short_amt.tolist()))
        added = []
        # candidate aisles: union of aisles stocking any short item, not already selected
        candidates = set()
        for it in remaining:
            for a_idx, _ in self.inst.item_to_aisles[it]:
                if a_idx not in self.sel_aisles:
                    candidates.add(a_idx)
        candidates = list(candidates)
        while remaining and candidates:
            best_a, best_cov = None, -1
            for a_idx in candidates:
                idx, val = self.inst.aisle_items[a_idx]
                cov = 0
                for it, v in zip(idx.tolist(), val.tolist()):
                    if it in remaining:
                        cov += min(v, remaining[it])
                if cov > best_cov:
                    best_cov, best_a = cov, a_idx
            if best_a is None or best_cov <= 0:
                break
            self.add_aisle_raw(best_a)
            added.append(best_a)
            candidates.remove(best_a)
            idx, val = self.inst.aisle_items[best_a]
            for it, v in zip(idx.tolist(), val.tolist()):
                if it in remaining:
                    remaining[it] -= v
                    if remaining[it] <= 0:
                        del remaining[it]
        if remaining:
            # could not fully cover -> rollback
            for a_idx in added:
                self.remove_aisle_raw(a_idx)
            return None
        return added

    def free_fit_mask(self, candidate_idx):
        """Vectorized: which of candidate_idx orders can be added right now with
        zero new aisles (i.e. fully covered by current slack)?"""
        if len(candidate_idx) == 0:
            return np.empty(0, dtype=bool)
        sub = self.inst.order_mat[candidate_idx]
        slack = self.supply - self.demand
        indptr = sub.indptr
        n = len(candidate_idx)
        nonempty = np.diff(indptr) > 0
        rowmax = np.full(n, -1.0)
        if sub.data.size > 0:
            excess = sub.data.astype(np.int64) - slack[sub.indices]
            starts = indptr[:-1][nonempty]
            vals = np.maximum.reduceat(excess, starts)
            rowmax[nonempty] = vals
        return rowmax <= 0

    def add_free_batch(self, candidate_set):
        """Repeatedly flood-fill: add every order in candidate_set that fits for
        free, until a fixed point (or UB reached). Mutates state. Returns the
        set of orders actually added."""
        added_any = set()
        pool = np.array(sorted(candidate_set), dtype=np.int64)
        while pool.size > 0:
            mask = self.free_fit_mask(pool)
            if not mask.any():
                break
            fitting = pool[mask]
            # add biggest-first, respecting UB
            order_vals = self.inst.order_units[fitting]
            order_sort = fitting[np.argsort(-order_vals)]
            progressed = False
            keep = []
            for o in order_sort.tolist():
                units = int(self.inst.order_units[o])
                if self.total_units + units > self.inst.wave_ub:
                    continue
                # re-verify (slack may have shrunk within this pass)
                idx, val = self.inst.order_items[o]
                if np.all(self.supply[idx] - self.demand[idx] >= val):
                    self.add_order_raw(o)
                    added_any.add(o)
                    progressed = True
                else:
                    keep.append(o)
            pool = np.array(sorted(set(pool.tolist()) - added_any), dtype=np.int64)
            if not progressed:
                break
        return added_any

    def dry_run_new_aisles(self, o):
        """Estimate (without mutating) the aisles that would need to be added
        to cover order o's shortfall, via greedy weighted set cover."""
        short_idx, short_amt = self.shortfall_for_order(o)
        if len(short_idx) == 0:
            return []
        remaining = dict(zip(short_idx.tolist(), short_amt.tolist()))
        added = []
        candidates = set()
        for it in remaining:
            for a_idx, _ in self.inst.item_to_aisles[it]:
                if a_idx not in self.sel_aisles:
                    candidates.add(a_idx)
        candidates = list(candidates)
        guard = 0
        while remaining and candidates and guard < 50:
            guard += 1
            best_a, best_cov = None, -1
            for a_idx in candidates:
                idx, val = self.inst.aisle_items[a_idx]
                cov = 0
                for it, v in zip(idx.tolist(), val.tolist()):
                    if it in remaining:
                        cov += min(v, remaining[it])
                if cov > best_cov:
                    best_cov, best_a = cov, a_idx
            if best_a is None or best_cov <= 0:
                return None
            added.append(best_a)
            candidates.remove(best_a)
            idx, val = self.inst.aisle_items[best_a]
            for it, v in zip(idx.tolist(), val.tolist()):
                if it in remaining:
                    remaining[it] -= v
                    if remaining[it] <= 0:
                        del remaining[it]
        if remaining:
            return None
        return added

    def try_add_order(self, o, q):
        """Try adding order o; if marginal value (under weight q) is positive, keep it.
        Returns True if added."""
        if o in self.sel_orders:
            return False
        units = int(self.inst.order_units[o])
        if self.total_units + units > self.inst.wave_ub:
            return False
        short_idx, short_amt = self.shortfall_for_order(o)
        if len(short_idx) == 0:
            # free addition, no new aisles needed
            self.add_order_raw(o)
            return True
        added = self.cover_shortfall_greedy(short_idx, short_amt)
        if added is None:
            return False
        marginal = units - q * len(added)
        if marginal > 1e-9:
            self.add_order_raw(o)
            return True
        # revert aisle additions, not worth it
        for a_idx in added:
            self.remove_aisle_raw(a_idx)
        return False

    def remove_redundant_aisles(self):
        """Remove any selected aisle whose removal keeps all demand covered."""
        changed = True
        while changed:
            changed = False
            for a_idx in list(self.sel_aisles):
                idx, val = self.inst.aisle_items[a_idx]
                new_supply = self.supply[idx] - val
                if np.all(new_supply >= self.demand[idx]):
                    self.remove_aisle_raw(a_idx)
                    changed = True

    def try_remove_order_if_frees_aisle(self, o, q):
        """Remove order o if doing so lets us drop >=1 aisle with net ratio-surrogate gain."""
        if o not in self.sel_orders:
            return False
        units = int(self.inst.order_units[o])
        if self.total_units - units < self.inst.wave_lb:
            return False
        idx, val = self.inst.order_items[o]
        trial_demand = self.demand.copy()
        trial_demand[idx] -= val
        freed = []
        for a_idx in list(self.sel_aisles):
            a_idx_items, a_val = self.inst.aisle_items[a_idx]
            new_supply = self.supply[a_idx_items] - a_val
            # would remaining aisles (minus this one) still cover trial_demand?
            if np.all(new_supply >= trial_demand[a_idx_items]):
                freed.append(a_idx)
        if len(freed) == 0:
            return False
        gain = q * len(freed) - units
        if gain > 1e-9:
            self.remove_order_raw(o)
            for a_idx in freed:
                self.remove_aisle_raw(a_idx)
            self.remove_redundant_aisles()
            return True
        return False


def construct2(inst, q, rng, deadline, sample_size=800):
    """Density-first construction: repeatedly (a) flood-fill every order that
    fits for free given current aisles, then (b) if room remains under UB,
    expand by the single order (from a random sample) whose marginal value
    units/new_aisles (or units - q*new_aisles) is best, committing its new
    aisles, and loop back to (a). This favors small, reusable aisle sets."""
    state = State(inst)
    remaining_pool = set(range(inst.O))

    # seed with a couple of random orders to kick off aisle selection
    seed_candidates = rng.sample(sorted(remaining_pool), min(5, len(remaining_pool)))
    for o in seed_candidates:
        if time.time() >= deadline:
            break
        added = state.dry_run_new_aisles(o)
        if added is None:
            continue
        units = int(inst.order_units[o])
        if state.total_units + units > inst.wave_ub:
            continue
        for a_idx in added:
            state.add_aisle_raw(a_idx)
        state.add_order_raw(o)
        remaining_pool.discard(o)
        break

    guard = 0
    while time.time() < deadline and state.total_units < inst.wave_ub:
        guard += 1
        if guard > 4 * inst.O + 1000:
            break
        added_free = state.add_free_batch(remaining_pool)
        remaining_pool -= added_free
        if time.time() >= deadline or state.total_units >= inst.wave_ub:
            break
        if not remaining_pool:
            break
        below_lb = state.total_units < inst.wave_lb
        sample = rng.sample(sorted(remaining_pool), min(sample_size, len(remaining_pool)))
        best_o, best_added, best_score = None, None, -1e18
        for o in sample:
            units = int(inst.order_units[o])
            if state.total_units + units > inst.wave_ub:
                continue
            added = state.dry_run_new_aisles(o)
            if added is None:
                continue
            n_new = len(added)
            score = units - q * n_new if n_new > 0 else float("inf")
            if score > best_score:
                best_score, best_o, best_added = score, o, added
        if best_o is None:
            break
        if best_score <= 0 and not below_lb:
            break
        for a_idx in best_added:
            if a_idx not in state.sel_aisles:
                state.add_aisle_raw(a_idx)
        state.add_order_raw(best_o)
        remaining_pool.discard(best_o)

    state.remove_redundant_aisles()
    return state


def construct(inst, q, rng, time_left_fn, rcl_frac=0.3):
    """GRASP construction for the surrogate objective units(X) - q*|Y|."""
    state = State(inst)
    order_idx = np.argsort(-inst.order_units)  # biggest orders first
    pool = order_idx.tolist()
    while pool:
        if time_left_fn() <= 0:
            break
        k = max(1, int(len(pool) * rcl_frac))
        rcl = pool[:k]
        o = rng.choice(rcl)
        pool.remove(o)
        state.try_add_order(o, q)
        if state.total_units >= inst.wave_lb and state.total_units >= inst.wave_ub * 0.98:
            break
    state.remove_redundant_aisles()
    return state


def trim_pass(state, inst, q):
    """Deterministic sweep: try to drop each selected order (smallest units
    first) if doing so frees at least one aisle under the current q weight."""
    for o in sorted(state.sel_orders, key=lambda o: inst.order_units[o]):
        if o in state.sel_orders:
            state.try_remove_order_if_frees_aisle(o, q)
    state.remove_redundant_aisles()
    return state


def local_search(state, inst, q, rng, deadline, max_fails=400):
    order_list = list(range(inst.O))
    fails = 0
    while time.time() < deadline and fails < max_fails:
        move = rng.random()
        improved = False
        if move < 0.5:
            o = rng.choice(order_list)
            if o not in state.sel_orders:
                improved = state.try_add_order(o, q)
            else:
                improved = state.try_remove_order_if_frees_aisle(o, q)
        else:
            o = rng.choice(order_list)
            if o in state.sel_orders:
                improved = state.try_remove_order_if_frees_aisle(o, q)
            else:
                improved = state.try_add_order(o, q)
        if improved:
            fails = 0
        else:
            fails += 1
    state.remove_redundant_aisles()
    return state


def ensure_lb(inst, state, rng, deadline):
    """If below the lower bound, force-add cheapest feasible orders until LB met."""
    order_idx = np.argsort(inst.order_units)  # smallest first (cheapest to add)
    i = 0
    guard = 0
    while state.total_units < inst.wave_lb and time.time() < deadline and guard < inst.O * 2:
        guard += 1
        progressed = False
        for o in order_idx:
            o = int(o)
            if o in state.sel_orders:
                continue
            units = int(inst.order_units[o])
            if state.total_units + units > inst.wave_ub:
                continue
            short_idx, short_amt = state.shortfall_for_order(o)
            added = state.cover_shortfall_greedy(short_idx, short_amt) if len(short_idx) else []
            if added is None:
                continue
            state.add_order_raw(o)
            progressed = True
            if state.total_units >= inst.wave_lb:
                break
        if not progressed:
            break
    return state


def solve_heuristic(inst, time_budget, seed=0):
    rng = random.Random(seed)
    deadline0 = time.time() + time_budget
    best_state = None
    best_ratio = -1.0

    # initial q guess: quick greedy with q=0 (ignore aisle cost) to estimate scale
    q = 0.0
    iteration = 0
    cur_state = None
    while time.time() < deadline0:
        iteration += 1
        remaining = deadline0 - time.time()
        if remaining <= 0:
            break
        # time slice for this Dinkelbach iteration
        slice_budget = max(1.0, remaining / 3) if iteration <= 3 else remaining
        iter_deadline = min(deadline0, time.time() + slice_budget)

        state = construct2(inst, q, rng, iter_deadline)

        state = local_search(state, inst, q, rng, iter_deadline)

        if state.total_units < inst.wave_lb:
            state = ensure_lb(inst, state, rng, iter_deadline)
            state.remove_redundant_aisles()

        state = trim_pass(state, inst, q)
        if state.total_units >= inst.wave_lb:
            state = trim_pass(state, inst, q)

        if state.sel_aisles and inst.wave_lb <= state.total_units <= inst.wave_ub:
            ratio = state.total_units / len(state.sel_aisles)
            if ratio > best_ratio:
                best_ratio = ratio
                best_state = state.clone()
            new_q = ratio
        else:
            new_q = q

        cur_state = state
        if abs(new_q - q) < 1e-6 and iteration > 1:
            q = new_q
            # perturb / restart with a fresh GRASP draw for diversification
            if time.time() < deadline0:
                state2 = construct2(inst, q, rng, deadline0)
                state2 = local_search(state2, inst, q, rng, deadline0)
                if state2.total_units < inst.wave_lb:
                    state2 = ensure_lb(inst, state2, rng, deadline0)
                    state2.remove_redundant_aisles()
                if state2.sel_aisles and inst.wave_lb <= state2.total_units <= inst.wave_ub:
                    r2 = state2.total_units / len(state2.sel_aisles)
                    if r2 > best_ratio:
                        best_ratio = r2
                        best_state = state2.clone()
            cur_state = None
            continue
        q = new_q

    if best_state is None:
        return [], [], 0.0
    return sorted(best_state.sel_orders), sorted(best_state.sel_aisles), best_ratio

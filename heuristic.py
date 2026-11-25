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
        """Remove order o if doing so lets us drop >=1 aisle with net ratio-surrogate gain.

        Aisles are dropped one at a time on a copy of the state. Two aisles that
        each cover the same item look redundant individually, and removing both
        at once would leave the remaining orders uncovered.
        """
        if o not in self.sel_orders:
            return False
        units = int(self.inst.order_units[o])
        if self.total_units - units < self.inst.wave_lb:
            return False
        trial = self.clone()
        trial.remove_order_raw(o)
        n_before = len(trial.sel_aisles)
        trial.remove_redundant_aisles()
        n_freed = n_before - len(trial.sel_aisles)
        if n_freed <= 0 or not trial.sel_aisles:
            return False
        if q * n_freed - units <= 1e-9:
            return False
        self.demand = trial.demand
        self.supply = trial.supply
        self.sel_orders = trial.sel_orders
        self.sel_aisles = trial.sel_aisles
        self.total_units = trial.total_units
        return True


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
    """One pass of add-or-remove moves. The coin flip chooses the move:
    add an order that is out of the wave, or drop one that is in it.
    """
    order_list = list(range(inst.O))
    fails = 0
    while time.time() < deadline and fails < max_fails:
        improved = False
        if rng.random() < 0.5:
            for _ in range(20):
                o = rng.choice(order_list)
                if o not in state.sel_orders:
                    improved = state.try_add_order(o, q)
                    break
        elif state.sel_orders:
            o = rng.choice(tuple(state.sel_orders))
            improved = state.try_remove_order_if_frees_aisle(o, q)
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


def _aisle_pool(inst, k):
    """Aisles worth trying: those that fully cover many units alone, interleaved
    with those that hold a lot of stock that the backlog actually demands."""
    full = np.zeros(inst.A, dtype=np.float64)
    for o in range(inst.O):
        idx, val = inst.order_items[o]
        possible = None
        for it, v in zip(idx.tolist(), val.tolist()):
            cands = [a for a, qty in inst.item_to_aisles[it] if qty >= int(v)]
            if not cands:
                possible = set()
                break
            got = set(cands)
            possible = got if possible is None else possible & got
            if not possible:
                break
        if not possible:
            continue
        u = float(inst.order_units[o])
        for a in possible:
            full[a] += u
    np.minimum(full, inst.wave_ub, out=full)

    total_demand = np.asarray(inst.order_mat.sum(axis=0)).ravel()
    useful = np.zeros(inst.A, dtype=np.float64)
    for a, (idx, val) in enumerate(inst.aisle_items):
        if len(idx) == 0:
            continue
        useful[a] = float(np.minimum(val, total_demand[idx]).sum())

    by_full = np.argsort(-full)
    by_useful = np.argsort(-useful)
    pool = []
    seen = set()
    for i in range(inst.A):
        for src, score in ((by_full, full), (by_useful, useful)):
            a = int(src[i])
            if a in seen or score[a] <= 0:
                continue
            seen.add(a)
            pool.append(a)
            if len(pool) >= k:
                return pool
    return pool


def _subset_closest_to_cap(feas, units, cap):
    """0-1 subset of `feas` whose unit sum is as large as possible without passing `cap`.
    Returns the chosen order ids, or None when an exact subset-sum is too large."""
    n = len(feas)
    cap = int(cap)
    if n == 0 or cap <= 0:
        return []
    if n * cap > 8_000_000:
        return None
    reachable = np.zeros(cap + 1, dtype=bool)
    reachable[0] = True
    prev_i = np.full(cap + 1, -1, dtype=np.int32)
    prev_s = np.full(cap + 1, -1, dtype=np.int32)
    for i, u in enumerate(units.tolist()):
        u = int(u)
        if u <= 0 or u > cap:
            continue
        src = np.flatnonzero(reachable[: cap + 1 - u])
        dest = src + u
        mask = ~reachable[dest]
        if not mask.any():
            continue
        src = src[mask]
        dest = dest[mask]
        reachable[dest] = True
        prev_i[dest] = i
        prev_s[dest] = src
    best_s = int(np.max(np.flatnonzero(reachable)))
    chosen = []
    s = best_s
    while s > 0 and prev_i[s] >= 0:
        i = int(prev_i[s])
        chosen.append(int(feas[i]))
        s = int(prev_s[s])
    return chosen


def pack_aisles(inst, aisles):
    """Fill a fixed aisle set. With |Y| fixed, the ratio is maximized by taking
    as many units as possible inside [LB, UB] without exceeding supply.
    Returns a feasible State, or None."""
    base = State(inst)
    for a in aisles:
        base.add_aisle_raw(int(a))
    if not base.sel_aisles:
        return None
    idx = np.arange(inst.O, dtype=np.int64)
    feas = idx[base.free_fit_mask(idx)]
    if feas.size == 0:
        return None

    total_dem = np.asarray(inst.order_mat[feas].sum(axis=0)).ravel()
    total_units = int(inst.order_units[feas].sum())
    no_conflict = bool(np.all(total_dem <= base.supply))

    def finish(chosen):
        st = base.clone()
        for o in chosen:
            u = int(inst.order_units[o])
            if st.total_units + u > inst.wave_ub:
                continue
            st.add_order_raw(int(o))
        st.remove_redundant_aisles()
        if not st.sel_aisles or st.total_units > inst.wave_ub:
            return None
        return st

    if no_conflict and total_units <= inst.wave_ub:
        return finish(feas.tolist())

    candidates = []
    if no_conflict and total_units > inst.wave_ub:
        picked = _subset_closest_to_cap(feas, inst.order_units[feas], inst.wave_ub)
        if picked is not None:
            candidates.append(picked)

    units = inst.order_units[feas]
    for sign in (-1.0, 1.0):
        order = feas[np.argsort(sign * units)]
        chosen = []
        dem = np.zeros(inst.I, dtype=np.int64)
        got = 0
        for o in order.tolist():
            u = int(inst.order_units[o])
            if got + u > inst.wave_ub:
                continue
            iix, val = inst.order_items[o]
            if np.all(base.supply[iix] - dem[iix] >= val):
                dem[iix] += val
                got += u
                chosen.append(int(o))
        candidates.append(chosen)

    best = None
    best_key = None
    for chosen in candidates:
        st = finish(chosen)
        if st is None:
            continue
        feasible = inst.wave_lb <= st.total_units <= inst.wave_ub
        # A wave inside the bounds beats any partial fill. Among feasible
        # waves the ratio decides; below LB, more units is closer to a wave.
        key = (feasible, st.total_units / len(st.sel_aisles) if feasible else st.total_units)
        if best_key is None or key > best_key:
            best_key = key
            best = st
    return best


def solve_aisle_search(inst, time_budget, seed=0):
    """Search small aisle sets, then pack each one. Best-known solutions visit
    only a handful of aisles, so the ratio moves much more with Y than with
    the order in which orders are appended."""
    if time_budget <= 0:
        return [], [], 0.0
    deadline = time.time() + time_budget
    if inst.O > 20000:
        pool_n, parents_n, width, max_size = 12, 8, 2, 4
    elif inst.O > 8000:
        pool_n, parents_n, width, max_size = 16, 10, 3, 6
    else:
        pool_n, parents_n, width, max_size = 20, 12, 4, 15

    wide_n = 40 if inst.O > 8000 else 80
    wide = _aisle_pool(inst, max(pool_n, wide_n))
    pool = wide[:pool_n]
    best_state = None
    best_ratio = -1.0
    cache = {}

    def evaluate(aisle_list):
        """Pack this aisle set. Returns (beam_score, state). Feasible waves
        always outrank partial fills in the beam."""
        nonlocal best_state, best_ratio
        key = tuple(sorted(int(a) for a in aisle_list))
        if key in cache:
            return cache[key]
        if time.time() >= deadline:
            return None
        st = pack_aisles(inst, key)
        if st is None or not st.sel_aisles:
            cache[key] = (0.0, None)
            return cache[key]
        feasible = inst.wave_lb <= st.total_units <= inst.wave_ub
        if feasible:
            ratio = st.total_units / len(st.sel_aisles)
            score = ratio
            if ratio > best_ratio:
                best_ratio = ratio
                best_state = st
        else:
            score = -1e18 + st.total_units
        cache[key] = (score, st)
        return cache[key]

    # A single aisle is often below LB. Grow those sets anyway.
    ranked = []
    for a in pool[:parents_n]:
        if time.time() >= deadline:
            break
        got = evaluate((a,))
        if got is None:
            continue
        ranked.append((got[0], (a,)))
    beam = sorted(ranked, key=lambda t: -t[0])

    for _size in range(2, max_size + 1):
        if time.time() >= deadline or not beam:
            break
        nxt = []
        seen = set()
        expand = beam if _size == 2 else beam[:width]
        for _score, aisles in expand:
            have = set(aisles)
            for a in pool:
                if time.time() >= deadline:
                    break
                if a in have:
                    continue
                key = tuple(sorted(have | {a}))
                if key in seen:
                    continue
                seen.add(key)
                got = evaluate(key)
                if got is None:
                    continue
                nxt.append((got[0], key))
        if not nxt:
            break
        nxt.sort(key=lambda t: -t[0])
        keep = parents_n if _size == 2 else width
        beam = nxt[:keep]

    # Prefixes of the ranked pool catch waves that need more aisles than the
    # beam reaches. Deleting and swapping then drops aisles that dilute the ratio.
    for k in range(1, min(len(wide), 30) + 1):
        if time.time() >= deadline:
            break
        evaluate(wide[:k])

    passes = 0
    while best_state is not None and time.time() < deadline and passes < 8:
        passes += 1
        before = best_ratio
        base = list(best_state.sel_aisles)
        improved = False
        for a in base:
            if time.time() >= deadline:
                break
            trial = [x for x in base if x != a]
            if not trial:
                continue
            got = evaluate(trial)
            if got is not None and got[0] > before + 1e-6:
                improved = True
                break
        if improved:
            continue
        have = set(best_state.sel_aisles)
        outside = [a for a in wide if a not in have]
        for out in list(best_state.sel_aisles):
            if improved or time.time() >= deadline:
                break
            for inn in outside:
                if time.time() >= deadline:
                    break
                trial = [x for x in best_state.sel_aisles if x != out] + [inn]
                got = evaluate(trial)
                if got is not None and got[0] > before + 1e-6:
                    improved = True
                    break
        if not improved:
            break

    if best_state is None:
        return [], [], 0.0
    return sorted(best_state.sel_orders), sorted(best_state.sel_aisles), best_ratio


def solve_heuristic(inst, time_budget, seed=0):
    rng = random.Random(seed)
    deadline0 = time.time() + time_budget
    best_state = None
    best_ratio = -1.0

    aisle_cut = time.time() + time_budget * 0.5
    orders, aisles, ratio = solve_aisle_search(inst, max(0.0, aisle_cut - time.time()), seed=seed)
    if aisles and ratio > best_ratio:
        best_ratio = ratio
        best_state = State(inst)
        for a in aisles:
            best_state.add_aisle_raw(a)
        for o in orders:
            best_state.add_order_raw(o)

    # initial q guess: aisle-set ratio when we have one, otherwise ignore aisle cost
    q = best_ratio if best_ratio > 0 else 0.0
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

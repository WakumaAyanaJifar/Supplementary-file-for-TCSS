"""
Escalation study for the minimal normative core (the norm-conflict task of the
proposed CyberLogicBench specification): auditable escalation (Rule R9) versus
silent tie-break baselines under precedence orders of varying completeness.

MODEL (base configuration)
--------------------------
* A norm set has N = 12 norms. Norm n has a latent stake mu_n ~ U(0,1), a
  prescribed binary action a_n in {0,1}, a creation time t_n and a specificity
  s_n in {1..4}. t_n and s_n are drawn independently of mu_n.
* The designer's precedence order is a WEAK order: norms are sorted by the noisy
  estimate est_n = mu_n + eps (eps ~ N(0, sigma_d)) and split into L equal-sized
  levels; n > m iff level(n) > level(m); norms on one level are incomparable.
  L = N gives a total order; L = 1 leaves every pair incomparable.
* In each event every norm is active independently with probability p_act. An
  active norm carries the per-event stake w_n = mu_n + eta, eta ~ N(0, sigma_e).
  Stakes are NOT clipped in the base configuration (a clipped variant is tested).
* REFERENCE DECISION (what an error is measured against): the action with the
  larger summed per-event stake over ALL active norms. It is computed from the
  event alone, before and independently of any policy, and is the same for every
  policy. It is a synthetic aggregate-stake criterion, not a claim about what is
  legally or ethically correct. It can disagree with the precedence order
  because (i) the order is built from noisy estimates (eps), (ii) the per-event
  stakes differ from the latent stakes (eta), and (iii) a strict order lets the
  top-ranked norms decide even when many lower-ranked norms jointly carry more
  stake. Ties have probability zero for continuous stakes (broken toward 0).
* If all active norms prescribe the same action there is no conflict; every
  policy acts identically and agrees with the reference.
* Otherwise all policies apply Rule R8 to comparable conflicts: norms below the
  highest active level are defeated by the top-level norms. If the top-level
  active norms agree, their action is taken automatically (an automated
  decision, possibly an error). If they disagree, the conflict is incomparable:
    - R9           : escalate to a human arbiter who returns the reference
                     action with probability h (otherwise the other action);
                     costs c minutes; actuation is blocked until resolution.
    - estimate     : "forced order": complete the order with the designer's own
                     noisy estimates (the top-level norm with the largest est wins).
    - random       : a top-level norm chosen uniformly at random wins.
    - recency      : the most recently created top-level norm wins.
    - specificity  : the most specific top-level norm wins (ties at random).
* Decision error = a decision (automated, tie-broken, or by the arbiter) that
  differs from the reference action. Errors are counted per 1,000 events.
* Escalations per avoided error = escalations / (errors under forced order minus
  errors under R9), as a ratio of means over norm sets, with a percentile
  bootstrap 95% CI over norm sets (2,000 resamples).
* Break-even arbiter accuracy h* = 1 - (E_forced - E_silent) / Esc, where E_silent
  is the number of automated errors under R9: the lowest accuracy at which R9
  makes no more errors than forced order.

REPRODUCIBILITY: all random streams are seeded; `python3 escalation_sim.py`
regenerates escalation_results.json exactly (tested: Python 3.12, NumPy 2.4).
"""
import json, sys
import numpy as np

BASE = dict(N=12, dist="uniform", p_one=0.5, ref="sum", sigma_d=0.15, sigma_e=0.10, clip=False)
LEVELS = {12: [12, 6, 4, 3, 2, 1], 24: [24, 12, 6, 4, 2, 1]}
H_ACC = 0.95
COST_MIN = 5.0
N_SETS = 300
N_SETS_SENS = 150
N_EVENTS = 1000
POLICIES = ["R9", "estimate", "random", "recency", "specificity"]
BOOT = 2000


def incomparable_fraction(L, n):
    k = n // L
    return L * k * (k - 1) / 2 / (n * (n - 1) / 2)


def make_norm_set(rng, L, c):
    N = c["N"]
    if c["dist"] == "uniform":
        mu = rng.uniform(0, 1, N)
    elif c["dist"] == "beta22":
        mu = rng.beta(2, 2, N)
    elif c["dist"] == "exp":
        mu = rng.exponential(0.5, N)
    else:
        raise ValueError(c["dist"])
    act = rng.integers(0, 2, N) if c["p_one"] == 0.5 else (rng.random(N) < c["p_one"]).astype(int)
    t = rng.permutation(N)
    spec = rng.integers(1, 5, N)
    est = mu + rng.normal(0, c["sigma_d"], N)
    order = np.argsort(est)
    level = np.empty(N, dtype=int)
    k = N // L
    for rank, idx in enumerate(order):
        level[idx] = rank // k
    return mu, act, t, spec, level, est


def reference(rng, c, acts, w):
    if c["ref"] == "sum":
        return 1 if w[acts == 1].sum() > w[acts == 0].sum() else 0
    if c["ref"] == "max":
        return int(acts[np.argmax(w)])
    if c["ref"] == "count":
        c1, c0 = int((acts == 1).sum()), int((acts == 0).sum())
        return 1 if c1 > c0 else 0 if c0 > c1 else int(rng.integers(0, 2))
    raise ValueError(c["ref"])


def run_set(rng, L, p_act, h_acc, c):
    N = c["N"]
    mu, act, t, spec, level, est = make_norm_set(rng, L, c)
    out = {p: dict(esc=0, silent=0, human=0, conflict=0) for p in POLICIES}
    for _ in range(N_EVENTS):
        active = np.flatnonzero(rng.random(N) < p_act)
        if active.size == 0:
            continue
        acts = act[active]
        if acts.min() == acts.max():
            continue
        w = mu[active] + rng.normal(0, c["sigma_e"], active.size)
        if c["clip"]:
            w = np.maximum(w, 0.0)
        truth = reference(rng, c, acts, w)
        top = active[level[active] == level[active].max()]
        top_acts = act[top]
        for p in POLICIES:
            out[p]["conflict"] += 1
        if top_acts.min() == top_acts.max():
            decided = top_acts[0]
            for p in POLICIES:
                out[p]["silent"] += int(decided != truth)
            continue
        out["R9"]["esc"] += 1
        out["R9"]["human"] += int(rng.random() > h_acc)
        choice = {"estimate": act[top[np.argmax(est[top])]],
                  "random": act[rng.choice(top)],
                  "recency": act[top[np.argmax(t[top])]]}
        best = spec[top].max()
        choice["specificity"] = act[rng.choice(top[spec[top] == best])]
        for p in ("estimate", "random", "recency", "specificity"):
            out[p]["silent"] += int(choice[p] != truth)
    return out


def summarize(vals):
    a = np.asarray(vals, dtype=float)
    return round(a.mean(), 2), round(1.96 * a.std(ddof=1) / np.sqrt(a.size), 2)


def ratio_ci(esc, diff, seed):
    esc, diff = np.asarray(esc, float), np.asarray(diff, float)
    if diff.mean() <= 0 or esc.mean() == 0:
        return None
    rg = np.random.default_rng(seed)
    n = esc.size
    idx = rg.integers(0, n, (BOOT, n))
    num, den = esc[idx].mean(1), diff[idx].mean(1)
    ok = den > 0
    r = num[ok] / den[ok]
    return [round(esc.mean() / diff.mean(), 2), round(float(np.percentile(r, 2.5)), 2), round(float(np.percentile(r, 97.5)), 2)]


def experiment(p_act, seed, h_acc=H_ACC, c=BASE, n_sets=N_SETS):
    rng = np.random.default_rng(seed)
    rows = []
    for L in LEVELS[c["N"]]:
        per = {p: dict(esc=[], silent=[], human=[], total=[], minutes=[], conflict=[]) for p in POLICIES}
        for _ in range(n_sets):
            r = run_set(rng, L, p_act, h_acc, c)
            for p in POLICIES:
                per[p]["esc"].append(r[p]["esc"])
                per[p]["silent"].append(r[p]["silent"])
                per[p]["human"].append(r[p]["human"])
                per[p]["total"].append(r[p]["silent"] + r[p]["human"])
                per[p]["minutes"].append(r[p]["esc"] * COST_MIN)
                per[p]["conflict"].append(r[p]["conflict"])
        row = dict(levels=L, incomparable=round(incomparable_fraction(L, c["N"]), 3))
        for p in POLICIES:
            row[p] = {k: summarize(v) for k, v in per[p].items()}
        diff = np.array(per["estimate"]["total"]) - np.array(per["R9"]["total"])
        row["esc_per_avoided"] = ratio_ci(per["R9"]["esc"], diff, seed + L)
        esc_m = np.mean(per["R9"]["esc"])
        row["breakeven_h"] = None if esc_m == 0 else round(1 - (np.mean(per["estimate"]["total"]) - np.mean(per["R9"]["silent"])) / esc_m, 3)
        rows.append(row)
    return rows


VARIANTS = [
    ("base", {}),
    ("stakes ~ Beta(2,2)", dict(dist="beta22")),
    ("stakes ~ Exp(mean 0.5)", dict(dist="exp")),
    ("reference = highest single stake", dict(ref="max")),
    ("reference = majority of active norms", dict(ref="count")),
    ("stakes clipped at 0", dict(clip=True)),
    ("skewed actions (P[a=1] = 0.2)", dict(p_one=0.2)),
    ("accurate designer (sigma_d = 0.05)", dict(sigma_d=0.05)),
    ("inaccurate designer (sigma_d = 0.30)", dict(sigma_d=0.30)),
    ("noisier events (sigma_e = 0.30)", dict(sigma_e=0.30)),
    ("24 norms", dict(N=24)),
]


def sensitivity():
    out = {}
    for i, (name, ov) in enumerate(VARIANTS):
        c = dict(BASE); c.update(ov)
        out[name] = dict(overrides=ov, rows=experiment(0.25, 20261001 + i, H_ACC, c, N_SETS_SENS))
    return out


MAIN_RUNS = {"0.15": dict(p_act=0.15, seed=20260929, h=H_ACC),
             "0.25": dict(p_act=0.25, seed=20260930, h=H_ACC),
             "0.35": dict(p_act=0.35, seed=20260931, h=H_ACC),
             # same seed as the p_act=0.25 run: identical norm sets and events, only arbiter accuracy differs
             "0.25_h0.8": dict(p_act=0.25, seed=20260930, h=0.8)}


def config():
    return dict(BASE, levels=LEVELS[12], human_accuracy=H_ACC, cost_minutes=COST_MIN, norm_sets=N_SETS,
                events_per_set=N_EVENTS, sensitivity_norm_sets=N_SETS_SENS, bootstrap_resamples=BOOT,
                seeds={k: v["seed"] for k, v in MAIN_RUNS.items()},
                sensitivity_seeds={name: 20261001 + i for i, (name, _) in enumerate(VARIANTS)},
                software=dict(python=sys.version.split()[0], numpy=np.__version__))


def run_main(key):
    m = MAIN_RUNS[key]
    return experiment(m["p_act"], m["seed"], m["h"])


def run_sens(i):
    name, ov = VARIANTS[i]
    c = dict(BASE); c.update(ov)
    return name, dict(overrides=ov, rows=experiment(0.25, 20261001 + i, H_ACC, c, N_SETS_SENS))


def report(results):
    for key, rows in results.items():
        print(f"\np_act/config={key}")
        for r in rows:
            print(r["incomparable"], "esc", r["R9"]["esc"][0], "| R9", r["R9"]["total"][0], "| forced", r["estimate"]["total"][0],
                  "| tiebreak", r["random"]["total"][0], "| esc/avoided", r["esc_per_avoided"], "| h*", r["breakeven_h"])


if __name__ == "__main__":
    # Usage:  python3 escalation_sim.py              -> everything, writes escalation_results.json
    #         python3 escalation_sim.py main <key>   -> one main configuration -> part_main_<key>.json
    #         python3 escalation_sim.py sens <i>     -> one sensitivity variant -> part_sens_<i>.json
    #         python3 escalation_sim.py merge        -> merge saved parts -> escalation_results.json
    args = sys.argv[1:]
    if args and args[0] == "main":
        json.dump(run_main(args[1]), open(f"part_main_{args[1]}.json", "w"))
    elif args and args[0] == "sens":
        json.dump(run_sens(int(args[1])), open(f"part_sens_{args[1]}.json", "w"))
    else:
        if args and args[0] == "merge":
            results = {k: json.load(open(f"part_main_{k}.json")) for k in MAIN_RUNS}
            sens = dict(json.load(open(f"part_sens_{i}.json")) for i in range(len(VARIANTS)))
        else:
            results = {k: run_main(k) for k in MAIN_RUNS}
            sens = dict(run_sens(i) for i in range(len(VARIANTS)))
        with open("escalation_results.json", "w") as f:
            json.dump(dict(config=config(), results=results, sensitivity=sens), f, indent=1)
        report(results)

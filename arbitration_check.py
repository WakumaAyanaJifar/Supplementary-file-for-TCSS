"""
Executable check of the arbitration semantics of the minimal normative core
(Rules R8/R9 and the effective relation) against Algorithm S1.

Abstraction used for testing: one actuator; each active norm n prescribes one
action a(n). Two norms CONFLICT iff they prescribe different actions (this is
what R_n(w) ∩ R_m(w) = ∅ amounts to under the exclusivity axiom). Precedence is
a random strict partial order (transitive closure of a random DAG).

semantic():   revised definition. Process norms top-down along a linear
              extension of the precedence order.
                m is DEFEATED  iff some active n ≻ m that is itself not defeated
                               conflicts with m;
                n, m are SUSPENDED iff both are undefeated, conflict, and are
                               incomparable;
                S = active − defeated − suspended;  admitted actions = actions of S.
algorithm_old(): a naive variant for comparison (pairs processed in an
              arbitrary order, candidates removed immediately).
algorithm_new(): Algorithm S1 (linear-extension order, then suspension).
"""
import itertools, random

def random_instance(rng, n, n_actions, p_edge):
    norms = list(range(n))
    perm = norms[:]; rng.shuffle(perm)            # hidden ranking used to keep edges acyclic
    pos = {x: i for i, x in enumerate(perm)}
    above = set()
    for a, b in itertools.permutations(norms, 2):
        if pos[a] < pos[b] and rng.random() < p_edge:
            above.add((a, b))                      # a ≻ b
    changed = True
    while changed:                                 # transitive closure
        changed = False
        for (a, b), (c, d) in itertools.product(list(above), repeat=2):
            if b == c and (a, d) not in above:
                above.add((a, d)); changed = True
    act = {x: rng.randrange(n_actions) for x in norms}
    return norms, above, act

def linear_extension(norms, above):
    out, rest = [], set(norms)
    while rest:
        m = next(x for x in sorted(rest) if not any((y, x) in above for y in rest))
        out.append(m); rest.remove(m)
    return out

def semantic(norms, above, act):
    order = linear_extension(norms, above)
    defeated = set()
    for m in order:
        if any((n, m) in above and n not in defeated and act[n] != act[m] for n in norms):
            defeated.add(m)
    und = [x for x in norms if x not in defeated]
    susp = set()
    for n, m in itertools.combinations(und, 2):
        if act[n] != act[m] and (n, m) not in above and (m, n) not in above:
            susp |= {n, m}
    S = [x for x in und if x not in susp]
    return {act[x] for x in S}, defeated, susp

def algorithm_old(norms, above, act, rng):
    cand = set(norms); D = set()
    pairs = [(a, b) for a, b in itertools.combinations(norms, 2) if act[a] != act[b]]
    rng.shuffle(pairs)                             # pair order is unspecified in the pseudocode
    for n, m in pairs:
        if n not in cand or m not in cand:
            continue
        if (n, m) in above:   D.add(m); cand.discard(m)
        elif (m, n) in above: D.add(n); cand.discard(n)
        else:                 cand.discard(n); cand.discard(m)
    return {act[x] for x in cand}

def algorithm_new(norms, above, act):
    order = linear_extension(norms, above); D = set()
    for m in order:
        if any((n, m) in above and n not in D and act[n] != act[m] for n in norms):
            D.add(m)
    und = [x for x in norms if x not in D]; susp = set()
    for n, m in itertools.combinations(und, 2):
        if act[n] != act[m] and (n, m) not in above and (m, n) not in above:
            susp |= {n, m}
    return {act[x] for x in und if x not in susp}

if __name__ == "__main__":
    rng = random.Random(20261010)
    T = 20000; bad_old = bad_new = 0; surv_conflict = 0; example = None
    for _ in range(T):
        n = rng.randint(2, 7); k = rng.randint(2, 3)
        norms, above, act = random_instance(rng, n, k, rng.choice([0.2, 0.5, 0.8]))
        sem, D, susp = semantic(norms, above, act)
        S = [x for x in norms if x not in D and x not in susp]
        surv_conflict += any(act[a] != act[b] for a, b in itertools.combinations(S, 2))
        old = algorithm_old(norms, above, act, rng); new = algorithm_new(norms, above, act)
        if old != sem:
            bad_old += 1
            if example is None and len(norms) <= 4: example = (norms, sorted(above), act, sem, old)
        bad_new += (new != sem)
    print(f"random instances: {T}")
    print(f"survivors that conflict pairwise (should be 0): {surv_conflict}")
    print(f"naive pair-order variant disagrees with the semantics: {bad_old} ({100*bad_old/T:.1f}%)")
    print(f"Algorithm S1 disagrees with the semantics: {bad_new}")
    print("smallest counterexample to the naive variant (norms, precedence, actions, semantic admit, variant admit):", example)

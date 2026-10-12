"""
Recomputes every simulation and corpus number quoted in the manuscript from
escalation_results.json and corpus_coding.csv, and checks that the manuscript
(main_TCSS.tex) states them. Exits with status 1 if any check fails.
Usage:  python3 verify_claims.py [main_TCSS.tex]
"""
import csv, json, re, sys
from decimal import Decimal, ROUND_HALF_UP
import numpy as np

def hu(x, nd):
    """Round half up (the convention used in the text), not Python's binary rounding."""
    return float(Decimal(str(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


main_path = sys.argv[1] if len(sys.argv) > 1 else "main_TCSS.tex"
M = open(main_path).read()
R = json.load(open("escalation_results.json"))
res, sens = R["results"], R["sensitivity"]
rows = list(csv.DictReader(open("corpus_coding.csv")))
inc = [r for r in rows if r["status"] == "included"]
fails = []

def check(name, cond, detail=""):
    print(("ok    " if cond else "FAIL  ") + name + (("  " + detail) if detail else ""))
    if not cond:
        fails.append(name)

def has(s): return s in M
f0 = lambda x: "%.0f" % x

# ---------------- corpus counts
dims = {d: sum(1 for r in inc if r["primary_dimension"] == d) for d in "IPXC"}
types = {t: sum(1 for r in inc if r["study_type"] == t) for t in "RCFE"}
nonrev = {d: sum(1 for r in inc if r["primary_dimension"] == d and r["study_type"] != "R") for d in "IPXC"}
check("60 included studies", len(inc) == 60)
check("dimension counts 19/20/17/4 and shares", (dims["P"], dims["X"], dims["I"], dims["C"]) == (19, 20, 17, 4)
      and all(has(s) for s in ["19 Policy, Norm, and Ethics", "20 Cross-Dimension", "17 Interaction-Driven", "4 Context-Aware", "31.7", "33.3", "28.3", "6.7"]))
check("type counts 20/19/18/3", (types["F"], types["R"], types["C"], types["E"]) == (20, 19, 18, 3) and has("20 studies present a formalism or working system, 19 are reviews, 18 are conceptual or position papers, and 3 are empirical"))
check("non-review counts 14/12/12/3 (41)", (nonrev["P"], nonrev["I"], nonrev["X"], nonrev["C"]) == (14, 12, 12, 3) and has("41 other studies") and has("14 policy, norm, and ethics, 12 interaction-driven, 12 cross-dimension, and 3 context-aware"))
xr = sum(1 for r in inc if r["primary_dimension"] == "X" and r["study_type"] in "RC")
check("16 of 20 integration studies are reviews or conceptual; 4 formal", xr == 16 and has("16 of those 20"))
check("37 reviews or conceptual", types["R"] + types["C"] == 37 and has("37 of the 60 studies"))
check("final_id numbering 1..60", sorted(int(r["final_id"]) for r in inc) == list(range(1, 61)))

# ---------------- PRISMA arithmetic
check("PRISMA arithmetic", 69 + 1025 + 315 + 235 + 500 + 16 == 2160 and 2160 - 300 == 1860 and 1860 - 1550 == 310
      and 58 + 1 + 84 + 104 + 3 == 250 and 310 - 250 == 60 and 69 + 1025 + 315 + 235 + 1340 == 2984)

# ---------------- simulation: Table VII (p_act = 0.25, h = 0.95)
base = res["0.25"]
tab = re.findall(r"(\d+)\\% & (\d+)(?:\$\\pm\$\d+)? & ([\d.]+) & (\d+)\$\\pm\$\d+ & (\d+) & (\d+) & ([\d.\-]+) \\\\", M)
check("Table VII has 6 rows", len(tab) == 6)
for (inc_, esc, hrs, r9, forced, tie, ratio), r in zip(tab, base):
    ok = (esc == f0(r["R9"]["esc"][0]) and abs(float(hrs) - r["R9"]["esc"][0] * 5 / 60) < 0.05
          and r9 == f0(r["R9"]["total"][0]) and forced == f0(r["estimate"]["total"][0]) and tie == f0(r["random"]["total"][0])
          and (ratio == "--" or abs(float(ratio) - r["esc_per_avoided"][0]) < 0.06))
    check("Table VII row %s%%" % inc_, ok, "paper: esc %s hrs %s R9 %s forced %s tie %s ratio %s" % (esc, hrs, r9, forced, tie, ratio))

# ---------------- simulation: quoted statements
conf = [r["R9"]["conflict"][0] for r in base]
check("62--63% of events have a conflict", min(conf) >= 620 and max(conf) <= 635 and has("62--63"), "range %.0f-%.0f" % (min(conf), max(conf)))
def tolerable(key):
    rows_ = res[key]; x = np.array([100 * r["incomparable"] for r in rows_]); y = np.array([r["R9"]["esc"][0] for r in rows_])
    return float(np.interp(50, y, x))
t = [tolerable(k) for k in ("0.15", "0.25", "0.35")]
check("tolerable shares ~13%, 7%, 4.5%", abs(t[0] - 13) < 0.5 and abs(t[1] - 7) < 0.5 and abs(t[2] - 4.5) < 0.3 and has("13\\%, 7\\%, and 4.5\\%"), str([round(v, 1) for v in t]))
r9 = base[1]
check("9%: 132 forced, 135 tie, 107 R9", (f0(r9["estimate"]["total"][0]), f0(r9["random"]["total"][0]), f0(r9["R9"]["total"][0])) == ("132", "135", "107") and has("132 under forced ordering and 135 under silent tie-breaks to 107"))
check("5.2 and 51.9 person-hours", f0(10 * base[1]["R9"]["esc"][0] * 5 / 60) == "52" and abs(base[1]["R9"]["esc"][0] * 5 / 60 - 5.2) < 0.05 and abs(base[5]["R9"]["esc"][0] * 5 / 60 - 51.9) < 0.05 and has("5.2 person-hours") and has("51.9 hours"))
ci1, ci5 = base[1]["esc_per_avoided"], base[5]["esc_per_avoided"]
check("ratio 2.5 [2.4,2.7] and 5.8 [5.6,6.1]", (hu(ci1[0], 1), hu(ci1[1], 1), hu(ci1[2], 1)) == (2.5, 2.4, 2.7) and (hu(ci5[0], 1), hu(ci5[1], 1), hu(ci5[2], 1)) == (5.8, 5.6, 6.1) and has("2.4--2.7") and has("5.6--6.1"), "%s %s" % (ci1, ci5))
h8 = res["0.25_h0.8"][-1]
check("h=0.8 at 100%: 125 vs 138, 623 escalations", (f0(h8["R9"]["total"][0]), f0(h8["estimate"]["total"][0]), f0(h8["R9"]["esc"][0])) == ("125", "138", "623") and has("125 errors against 138") and has("623 escalations"))
hs = [r["breakeven_h"] for r in base if r["breakeven_h"] is not None]
check("break-even 0.55 to 0.78 at p=0.25", (hu(hs[0], 2), hu(hs[-1], 2)) == (0.55, 0.78) and has("from 0.55 at 9") and has("0.78 at 100"), str([round(h, 3) for h in hs]))
allh = [r["breakeven_h"] for k in ("0.15", "0.25", "0.35") for r in res[k] if r["breakeven_h"] is not None]
check("break-even 0.54--0.80 over activation rates", (hu(min(allh), 2), hu(max(allh), 2)) == (0.54, 0.80) and has("0.54--0.80"), "%.3f-%.3f" % (min(allh), max(allh)))
sh = [r["breakeven_h"] for v in sens.values() for r in v["rows"] if r["breakeven_h"] is not None]
check("break-even 0.49--0.86 over variants", (hu(min(sh), 2), hu(max(sh), 2)) == (0.49, 0.86) and has("between 0.49 and 0.86"), "%.3f-%.3f" % (min(sh), max(sh)))
cells = [(r["R9"]["total"][0] < r["estimate"]["total"][0] and r["R9"]["total"][0] < r["random"]["total"][0]) for v in sens.values() for r in v["rows"] if r["R9"]["esc"][0] > 0]
check("R9 better in all variant cells", all(cells) and len(cells) == 55, "%d/%d" % (sum(cells), len(cells)))
check("ten variants plus base", len(sens) == 11 and has("In ten variants"))
ratios = [r["esc_per_avoided"][0] for k in ("0.15", "0.25", "0.35") for r in res[k] if r["esc_per_avoided"]]
check("ratio range over activation rates 2.5--6.6 (supplement)", (hu(min(ratios), 1), hu(max(ratios), 1)) == (2.5, 6.6), "%.2f-%.2f" % (min(ratios), max(ratios)))

# ---------------- abstract length
a = M[M.index("\\begin{abstract}") + 16:M.index("\\end{abstract}")]
words = len(re.sub(r"\\[a-zA-Z]+", "", re.sub(r"\$[^$]*\$", "X", a)).split())
check("abstract at most 250 words", words <= 250, str(words))

print("\n%d check(s) failed" % len(fails) if fails else "\nall checks passed")
sys.exit(1 if fails else 0)

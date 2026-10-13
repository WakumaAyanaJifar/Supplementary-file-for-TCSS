# Supplementary data and code

Data and code for the article *Logical Reasoning for Norms, Interaction, and Context in Computational Social Systems: A Systematic Review and a Minimal Normative Core*, submitted to *IEEE Transactions on Computational Social Systems*. The supplementary material of the article (Sections S-II-L and S-VII) describes these files.

| File | Content |
|---|---|
| `corpus_coding.csv` | Per-record coding of the 80 records that reached the taxonomy step: final study number (`final_id`, 1 to 60, matching the catalog in Supplementary S-III), original number, BibTeX key, year, short title, primary dimension (`I` interaction-driven, `C` context-aware, `P` policy/norm/ethics, `X` cross-dimension), study type (`R` review, `C` conceptual or position, `F` formal model or working system, `E` empirical), status, and notes. |
| `escalation_sim.py` | Simulation of auditable escalation (Rule R9) against forced ordering and silent tie-breaks. |
| `escalation_results.json` | Complete simulation output, including the sensitivity analysis. Every simulation number in the article and supplement comes from this file. |
| `verify_claims.py` | Recomputes every simulation and corpus number quoted in the article from the two data files and checks that the article states them. |
| `arbitration_check.py` | Tests Algorithm S1 against the definitions of conflict, defeat, and suspension on 20,000 random instances. |

Status values in `corpus_coding.csv`: `included` (the 60 primary studies), `excluded-c` and `excluded-c-first-pass` (the 17 records reclassified under criterion (c)), and `withheld-unverified` (the 3 records that could not be verified). Study types were assigned from titles, venues, and abstracts with language-model assistance and were not independently double-coded.

## Reproducing the results

Requirements: Python 3 and NumPy (tested with Python 3.12 and NumPy 2.4).

```bash
python3 escalation_sim.py                          # regenerates escalation_results.json exactly (about seven minutes)
python3 escalation_sim.py main 0.25                # one configuration only, saved as part_main_0.25.json
python3 verify_claims.py path/to/main_TCSS.tex     # checks every quoted number against the data
python3 arbitration_check.py                       # tests Algorithm S1
```

All random streams are seeded; the seeds are listed in the `config` section of `escalation_results.json` and in Supplementary S-VII.

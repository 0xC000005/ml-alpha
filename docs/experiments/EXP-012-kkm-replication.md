---
id: EXP-012
status: planned-gated
---

# EXP-012 — KKM (w33351) replication: protocol + 4-rung ladder (PLANNED, GATED)

**Framing correction (2026-07-11, from the user):** the replication target is **KKM** —
Kelly, Kuznetsov, Malamud & Xu (2025), *Artificial Intelligence Asset Pricing Models*,
NBER w33351 (in repo root) — **not GKX (2020)**. GKX is not a target at all. README/CLAUDE.md
still frame the project as a GKX replication; that framing is stale. KKM's statistic is the
**pooled OOS SDF-portfolio Sharpe**, not a decile long-short spread, so the decile /
value-weighting machinery is beside the point (`experiments/gkx_report.py` is misnamed; its
*pooling* logic is what matters).

**Where we actually stand vs KKM.** We are not failing to replicate it — we have not
attempted it. Every axis differs:

| axis | KKM | us (base arm today) |
|---|---|---|
| characteristics | 132 JKP (Jensen–Kelly–Pedersen) | 95 GKX-era signals |
| other inputs | characteristics only | **+8 macro +74 industry dummies** |
| feature transform | cross-sectional rank → **[−0.5, 0.5]**, missing→0 | pooled z-score, clip ±5σ |
| universe | shrcd 10–12, **drop nano** (<1st pct **NYSE** cap), drop rows missing >⅓ chars | no such filters |
| training window | **60-month rolling**, refit **every month** | expanding from 1975, refit yearly |
| OOS | Feb 1968 – Dec 2022, **659 months** | 2012–2019, **88 months** |
| penalty | nonlinear models **unpenalized** ("without a penalty term") | AdamW ridge on output head (`train_transformer_msrr.py:250`) |
| early stopping | none described — fixed epochs on the 60-month window | validation split + early stop (`:260`) |
| ensembling | 10 seeds, averaged | 10 seeds, **L1-normalized per seed per month** (`msrr_combine.py:28`) |
| model size | MLP >300K; transformer ~100K (1 blk) → ~1M (10 blk) | **14,369 params**, 1 block, d_model=32 |

Note the ensemble, penalty, early-stopping and L1-normalization rows: several are *our*
inventions (some of them good — L-02) that are **not in the paper**. A replication must
turn them off, or declare them.

**Ladder (KKM's own names — already in the glossary):** BSV (linear, own-asset) → linear
portfolio transformer (attention, still linear) → MLP (2×512 ReLU, own-asset, >300K) →
nonlinear portfolio transformer (attention + depth, 1→10 blocks). We currently have a toy
instance of the last rung **and no attention-free control under the MSRR loss**, so the
paper's central claim is presently **untestable in this repo**.

**The power situation — and the trap I nearly walked into.** Adopting the rolling window
buys ~**635** OOS months on our data (Feb 1968 – Dec 2020) instead of 88, cutting
SE(annualized Sharpe) from **0.38 → 0.14**. That is a real and large win. But it does *not*
make the headline claim testable:

| comparison | KKM's effect | ρ between arms (their Fig. 3) | MDE @80% power, T=635 | powered? |
|---|---|---|---|---|
| BSV → linear attention | **+0.3** (3.6→3.9) | 0.90 | ~0.18 | **YES** |
| MLP → transformer (attention, controlled) | **+0.26** (4.31→4.57) | 0.76 | ~0.27 | **NO — at the edge** |
| depth 1 blk → 10 blk | **+0.8** (3.8→4.6) | — | ~0.27 | **YES** |

My first draft of this plan claimed the nonlinear attention effect was **+0.7**. It is not:
+0.7 is transformer-vs-DKKM, which changes **depth *and* attention** and identifies neither.
The controlled comparison is transformer-vs-MLP = **+0.26**. (Caught by Codex; see L-11.)

**But the Δ-Sharpe table above is the WRONG TEST — and this rescues Stage 2.** KKM do not
test Sharpe *differences*. Their model-comparison statistic is a **spanning regression**,
eq. (26): `R_A,t = α + β·R_B,t + ε_t`, with both legs **rescaled to 15% annualized vol**,
reporting the alpha and its t-stat (this is where their "t = 6.8" comes from). That test
conditions on `R_B` and only has to detect the **residual** mean; at ρ = 0.76 the residual
vol is just 65% of total, so it is *far* more powerful than differencing two Sharpes:

| test | statistic at our levels (1.40 vs 1.14), T=635 | verdict |
|---|---|---|
| Δ-Sharpe (what Codex and I powered) | 80%-power MDE **0.28** vs effect **0.26** | underpowered |
| **KKM eq. (26) alpha regression** | **t(α) ≈ 6.0** (≈ **4.0** even with a 1.5× HAC haircut) | **detectable** |

(`t(α) = √T · √(SR²_tangency − SR²_B)` in monthly units — the appraisal ratio. At T=88 it is
only 2.2, which is why none of this was resolvable on the old window.) **Stage 2 is therefore
worth running — but only if the test is KKM's alpha regression with HAC errors, not a
Sharpe-difference test.** Powering the wrong estimator nearly killed a viable experiment; it
would also have licensed a meaningless one had the arithmetic gone the other way.

**Design (staged; each stage gates the next).**
- **Stage 0 — no GPU.** Rebuild the data layer to KKM spec (rank→[−0.5,0.5]; nano + missingness
  filters; **remove** the macro and industry channels — zeroing them is not removal, the bias
  term still learns a constant) and write a **rolling-window, monthly-refit** driver. Nothing
  in the repo does this: `train_transformer_msrr.py:366` is an explicit yearly *expanding*
  refit with an unbounded lower training mask (`:382`).
- **Stage 1 — BSV vs linear portfolio transformer.** Deterministic (closed-form/ridge + LOOCV),
  **no seeds** → ~635 solves, not 6,350 fits. Near-free, validates the entire protocol, and is
  **the one rung we are actually powered to test.** This is the checkpoint.
- **Stage 2 — MLP (2×512) vs 1-block transformer.** The controlled attention test. Expensive
  and, per the table above, **underpowered on 95 characteristics.** Do not launch until Stage 1
  passes and we have decided how to handle the power gap (see kill-bar).
- **Stage 3 — depth 1→10 blocks.** Powered (+0.8), but a full sweep is **63,500 nonlinear fits**.

**Kill-bars (written BEFORE the run).**
1. **Stage 1 must reproduce the *sign and rough size* of the linear attention effect (+0.3, and
   it is powered to).** If BSV ≥ linear-attention on our data, the apparatus or the data is
   wrong — stop and debug; do not proceed to any GPU stage.
2. **The attention claim is adjudicated by KKM's eq. (26) alpha regression (HAC errors), and
   the test is declared now, before the run.** A Sharpe-difference comparison is *not* the test
   and will not be quoted as one. Bar: **transformer vs MLP alpha t > 3** (Harvey–Liu–Zhu), both
   legs rescaled to 15% annualized vol. Expected t ≈ 6 if KKM's effect survives on our data
   (≈4 after a 1.5× HAC haircut) — so a null here is informative, not merely underpowered.
3. **Inference is HAC or block-bootstrap, not the iid Lo formula.** Adjacent models share 59/60
   training months, so weights (and hence OOS returns) are persistent; the realized returns
   don't overlap, but the iid SE is still not valid for the final number.
4. **The α must survive in BOTH directions.** Report α(A on B) *and* α(B on A) — KKM's own
   diagnostic (a real improvement shows a large α one way and ≈0 the other; :1255).

**Known blockers (found before launch, not after).**
- **The nano filter cannot be built from the data on disk.** It needs the **NYSE** market-cap
  1st percentile, and the parquets carry market cap but **no exchange identifier**. Either source
  `exchcd` from CRSP or declare a documented deviation (e.g. an all-universe percentile).
- Our data ends **2020-12**, KKM's 2022-12.
- The universe is thin early: ~1,102 stocks/mo in 1960 vs ~5,686 in 2012.
- **D-4 (universe look-ahead) is unresolved** and sits under all of this. It doesn't block a Δ,
  but a *level* Sharpe near 4.6 would deserve real scrutiny before anyone believes it.

**Cost:** Stage 0 free; Stage 1 ~CPU-hours; Stage 2 **~170–230 GPU-hours per nonlinear rung**;
Stage 3 up to **1,000+**. **GATED — nothing is submitted without an explicit go-ahead.**

---


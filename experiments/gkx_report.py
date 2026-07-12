#!/usr/bin/env python
"""Report portfolio performance the way Gu, Kelly & Xiu (2020) report it.

GKX Table 7 note, verbatim: "we report the performance of prediction-sorted portfolios
over the 30-year out-of-sample testing period. All stocks are sorted into deciles based
on their predicted returns for the next month. Columns 'Pred,' 'Avg,' 'SD,' and 'SR'
provide the predicted monthly returns for each decile, the average realized monthly
returns, their standard deviations, and Sharpe ratios, respectively. All portfolios are
value weighted."

So their statistic is ONE Sharpe over the POOLED out-of-sample monthly series,

    SR = mean(r_m) / sd(r_m) * sqrt(12),

on VALUE-weighted deciles. (Check against their own table: NN4 H-L is Avg 2.26%/mo,
SD 5.80%/mo, and 2.26/5.80*sqrt(12) = 1.35 = their printed SR.) Equal-weighted deciles
are Internet Appendix A.9, where Sharpes are far higher -- NN4 goes 1.35 -> 2.45.

This repo reports neither. collect_screen.py:40 takes a ROW-MEAN of per-year Sharpes --
a mean of ratios, biased upward -- on EQUAL-weighted deciles. Comparing that to GKX's
1.15 is comparing a microcap-heavy portfolio scored the flattering way against a
large-cap portfolio scored the honest way.

Two modes:

  # exact, no GPU, no retrain: pool per-year (mean, sd, n) from the summary CSVs
  gkx_report.py --summaries "output/repro/mse_*"
  gkx_report.py --summaries "output/exp/confirm_rank/base_*"

  # full report (EW *and* VW deciles, GKX-style decile table): needs per-stock preds
  gkx_report.py --predictions output/exp/confirm_rank/predictions.parquet

The --summaries mode can only recover the EQUAL-weighted number, because value weights
need per-stock predictions. See save_predictions() in the drivers.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

SQRT12 = np.sqrt(12.0)

# GKX (2020) Table 7 (value-weighted) and Internet Appendix A.9 (equal-weighted),
# long-short decile spread, 1987-2016 (360 months).
GKX = {
    "NN1 (VW)": 1.17, "NN2 (VW)": 1.16, "NN3 (VW)": 1.20,
    "NN4 (VW)": 1.35, "NN5 (VW)": 1.15, "NN4 (EW)": 2.45,
}


def se_sharpe(sr_annual: float, n_months: int) -> float:
    """Standard error of an ANNUALIZED Sharpe ratio (Lo 2002), annualized.

    SE = sqrt(12) * sqrt((1 + 0.5 * SR_monthly^2) / T).  The sqrt(12) is the whole
    ballgame: omitting it understates the noise by 3.46x, which is how this repo
    convinced itself a 1-year screen was informative (RESEARCH_LOG L-09).
    """
    sr_m = sr_annual / SQRT12
    return SQRT12 * np.sqrt((1.0 + 0.5 * sr_m ** 2) / n_months)


def report_series(name: str, r: np.ndarray, annual_srs=None) -> dict:
    """GKX statistic on a pooled monthly return series, + the biased stat for contrast."""
    T = len(r)
    mean, sd = r.mean(), r.std(ddof=1)
    sr = mean / sd * SQRT12
    se = se_sharpe(sr, T)
    out = {"name": name, "T": T, "mean_pct": mean * 100, "sd_pct": sd * 100,
           "sharpe": sr, "se": se, "t": sr / se}
    if annual_srs is not None and len(annual_srs):
        out["mean_of_annual"] = float(np.mean(annual_srs))
    return out


def print_rows(rows):
    print(f"{'portfolio':<28} {'T':>4} {'Avg %/mo':>9} {'SD %/mo':>8} "
          f"{'SR (GKX)':>9} {'SE':>6} {'t':>6}   {'mean-of-annual':>14}")
    print("-" * 96)
    for r in rows:
        moa = f"{r['mean_of_annual']:.2f}" if "mean_of_annual" in r else "--"
        bias = ""
        if "mean_of_annual" in r:
            bias = f"  (+{r['mean_of_annual'] - r['sharpe']:.2f} biased)"
        print(f"{r['name']:<28} {r['T']:>4} {r['mean_pct']:>9.3f} {r['sd_pct']:>8.3f} "
              f"{r['sharpe']:>9.2f} {r['se']:>6.2f} {r['t']:>6.2f}   {moa:>14}{bias}")


# --------------------------------------------------------------------------------------
# Mode 1: pool exactly from per-year summary CSVs (no predictions, no GPU, no retrain)
# --------------------------------------------------------------------------------------
def pool_from_summaries(pattern: str):
    """Reconstruct the pooled monthly moments from per-year (mean, sd, n).

    Exact, not an approximation -- law of total variance:
        M      = sum(n_y * m_y) / N
        Var_p  = sum(n_y * (v_y + (m_y - M)^2)) / N        [population]
    The drivers compute their per-year sd with np.std (ddof=0), so v_y is a population
    variance; we convert the pooled result to a sample variance at the end.

    This recovers the pooled Sharpe of the SAME (equal-weighted) portfolio that was
    already run. It cannot produce value weights -- that needs per-stock predictions.
    """
    dirs = sorted(glob.glob(pattern))
    frames = []
    for d in dirs:
        for f in glob.glob(os.path.join(d, "metrics", "*.csv")):
            frames.append(pd.read_csv(f))
    if not frames:
        sys.exit(f"no metrics CSVs under {pattern}/metrics/*.csv")
    D = pd.concat(frames, ignore_index=True).sort_values("test_year")

    # Which portfolio's moments did this run actually save?
    candidates = [
        ("decile L/S (equal-wt)", "mean_ls_ret_pct", "std_ls_ret_pct", "sharpe_ls_annual", 0.01),
        ("MSRR SDF portfolio", "sdf_mean_ret_l1", "sdf_std_ret_l1", "sdf_sharpe", 1.0),
        ("MSRR SDF portfolio", "sdf_mean_ret", "sdf_std_ret", "sdf_sharpe", 1.0),
    ]
    rows, found = [], False
    for label, mcol, scol, srcol, scale in candidates:
        if mcol not in D.columns or scol not in D.columns:
            continue
        sub = D.dropna(subset=[mcol, scol])
        if sub.empty:
            continue
        found = True
        m = sub[mcol].values * scale          # -> decimal monthly return
        v = (sub[scol].values * scale) ** 2   # population variance (ddof=0)
        n = sub["n_months"].values.astype(float)
        N = n.sum()
        M = (n * m).sum() / N
        var_p = (n * (v + (m - M) ** 2)).sum() / N
        sd = np.sqrt(var_p * N / (N - 1))     # -> sample sd

        sr = M / sd * SQRT12
        se = se_sharpe(sr, int(N))
        annual = sub[srcol].values if srcol in sub.columns else []
        rows.append({"name": label, "T": int(N), "mean_pct": M * 100, "sd_pct": sd * 100,
                     "sharpe": sr, "se": se, "t": sr / se,
                     "mean_of_annual": float(np.mean(annual)) if len(annual) else np.nan})

        # Self-check: the stored per-year Sharpe must equal mean/sd*sqrt(12) per year.
        # If this fails, the mean/sd columns are not the moments of the scored series.
        if len(annual):
            recomputed = (sub[mcol].values * scale) / (sub[scol].values * scale) * SQRT12
            err = np.abs(recomputed - annual).max()
            if err > 1e-4:
                print(f"  [warn] {label}: stored per-year SR disagrees with mean/sd by "
                      f"{err:.2e} -- reconstruction may not be exact", file=sys.stderr)

    if not found:
        sys.exit("These runs saved only a Sharpe ratio, not the monthly mean and sd, so "
                 "the pooled Sharpe CANNOT be reconstructed. Re-run with save_predictions.")

    print(f"\n=== POOLED (GKX convention) — {pattern} — {len(dirs)} run dirs ===\n")
    print_rows([r for r in rows if not np.isnan(r["sharpe"])])
    print("\nEqual-weighted only. GKX's headline Table 7 is VALUE weighted; for that you\n"
          "need per-stock predictions (--predictions).")


# --------------------------------------------------------------------------------------
# Mode 2: full report from per-stock predictions (EW + VW, GKX-style decile table)
# --------------------------------------------------------------------------------------
def load_mktcap() -> pd.DataFrame:
    """Market cap (mvel1) as long (month_id, permno) -> me.

    NOTE ON TIMING (RESEARCH_LOG L-09 #4): month_id is the FEATURE month t; the return
    is realized at t+1. The portfolio is FORMED at t, so the correct weight is the market
    cap at t -- known at formation, no look-ahead. We join on month_id directly.
    """
    me = pd.read_parquet("ml_alpha_data/gkx_full/mvel1.parquet")
    me.index = [int(str(p).replace("-", "")) for p in me.index]
    long = me.stack().rename("me").reset_index()
    long.columns = ["month_id", "permno", "me"]
    long["permno"] = long["permno"].astype(np.int64)
    return long[long["me"] > 0]


def decile_ls(df: pd.DataFrame, weight: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Monthly long-short decile-spread returns. weight=None -> equal, 'me' -> value.

    Decile construction follows the repo's existing convention (top n//10 vs bottom
    n//10 by predicted return), so the equal-weighted number is directly comparable to
    what compute_oos_metrics() has been producing.
    """
    months, rets = [], []
    for m, g in df.groupby("month_id", sort=True):
        n = len(g)
        k = n // 10
        if k < 2:
            continue
        g = g.iloc[np.argsort(g["pred"].values)]
        lo, hi = g.iloc[:k], g.iloc[-k:]
        if weight is None:
            r = hi["target"].mean() - lo["target"].mean()
        else:
            wh = hi[weight].values / hi[weight].values.sum()
            wl = lo[weight].values / lo[weight].values.sum()
            r = (hi["target"].values * wh).sum() - (lo["target"].values * wl).sum()
        months.append(m)
        rets.append(r)
    return np.array(months), np.array(rets)


def annual_sharpes(months: np.ndarray, rets: np.ndarray) -> list:
    """The biased statistic collect_screen.py reports, so we can show the gap."""
    years = months // 100
    return [rets[years == y].mean() / rets[years == y].std() * SQRT12
            for y in np.unique(years) if (years == y).sum() > 2]


def gkx_decile_table(df: pd.DataFrame, weight: str | None) -> pd.DataFrame:
    """GKX Table 7 layout: Pred / Avg / SD / SR for each decile, pooled over all months."""
    rows = {d: [] for d in range(10)}
    preds = {d: [] for d in range(10)}
    for m, g in df.groupby("month_id", sort=True):
        if len(g) < 20:
            continue
        g = g.iloc[np.argsort(g["pred"].values)]
        idx = np.minimum((np.arange(len(g)) * 10) // len(g), 9)
        for d in range(10):
            leg = g.iloc[idx == d]
            if leg.empty:
                continue
            if weight is None:
                rows[d].append(leg["target"].mean())
            else:
                w = leg[weight].values / leg[weight].values.sum()
                rows[d].append((leg["target"].values * w).sum())
            preds[d].append(leg["pred"].mean())
    out = []
    for d in range(10):
        r = np.array(rows[d])
        out.append({"decile": "Low(L)" if d == 0 else ("High(H)" if d == 9 else str(d + 1)),
                    "Pred": np.mean(preds[d]) * 100, "Avg": r.mean() * 100,
                    "SD": r.std(ddof=1) * 100, "SR": r.mean() / r.std(ddof=1) * SQRT12})
    return pd.DataFrame(out)


def full_report(pred_path: str):
    df = pd.read_parquet(pred_path)
    need = {"permno", "month_id", "pred", "target"}
    if not need.issubset(df.columns):
        sys.exit(f"{pred_path} must have columns {need}, has {set(df.columns)}")

    me = load_mktcap()
    merged = df.merge(me, on=["month_id", "permno"], how="left")
    cover = merged["me"].notna().mean()
    print(f"\nmarket-cap coverage: {cover:.1%} of stock-months "
          f"({merged['me'].isna().sum():,} unmatched, dropped from the VW leg only)")

    rows = []
    for label, sub, w in [("decile L/S (equal-wt)", merged, None),
                          ("decile L/S (VALUE-wt)", merged.dropna(subset=["me"]), "me")]:
        mo, r = decile_ls(sub, w)
        rows.append(report_series(label, r, annual_sharpes(mo, r)))

    print(f"\n=== POOLED (GKX convention) — {pred_path} ===\n")
    print_rows(rows)

    print("\n=== GKX Table 7 layout, VALUE-weighted (their headline) ===")
    print(gkx_decile_table(merged.dropna(subset=["me"]), "me").round(2).to_string(index=False))

    print("\n=== GKX reference (1987-2016, 360 months) ===")
    for k, v in GKX.items():
        print(f"    {k:<12} {v:.2f}")
    print("\nCompare like with like: our VALUE-weighted number vs their VW rows; our\n"
          "equal-weighted number vs NN4 (EW) = 2.45.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--summaries", help="glob of run dirs, e.g. 'output/repro/mse_*'")
    g.add_argument("--predictions", help="parquet with permno, month_id, pred, target")
    a = ap.parse_args()
    if a.summaries:
        pool_from_summaries(a.summaries)
    else:
        full_report(a.predictions)

"""
THE RACE - does the race add information beyond separation?  Plus threshold robustness.
----------------------------------------------------------------------------------------
1. Completion model. On pressured, targeted passes we fit a logistic regression (plain numpy, no sklearn)
   predicting completion from what a separation-based stat already knows - separation at the throw,
   how far the ball travelled (route depth), coverage type, down & distance, time to throw - and then add
   the race margin (seconds the target was open before the pocket broke). If AUC / log-loss improve
   out of sample, the race carries information that separation alone does not.
2. Robustness. Re-computes the headline numbers under different thresholds (pressure radius, open
   separation) on a 30-game subset, from the race_routes_<p>_<o>_<s>.csv files written by
   `RACE_PRESSURE=1.5 python compute_race.py --games 30` etc.

Writes output/model_results.csv, output/robustness.csv, output/chart_model.png.
Usage: python model.py
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
DATA = os.path.join(HERE, "nfl-big-data-bowl-regional-event-data-main", "data")
INK, MUTED, GRID = "#1f2630", "#6b7480", "#e3e7ec"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": .8})


# ----------------------------------------------------------------------------- data
def targeted(routes):
    """Rows of the receiver the QB threw to, pressured plays, C/I only (INT text differs)."""
    r = routes[routes.pressured & routes.passResult.isin(["C", "I"])].copy()
    def tgt(d):
        m = re.search(r"pass .*?to ([A-Z][A-Za-z'\-]*\.\s?[A-Za-z'\-\.]+)", str(d))
        return m.group(1).replace(" ", "").lower() if m else None
    r["tgt"] = r.playDescription.map(tgt)
    r["short"] = r.displayName.map(lambda n: f"{n.split()[0][0]}.{n.replace(' Jr.', '').replace(' III', '').replace(' II', '').split()[-1]}".lower())
    r = r[r.tgt.notna() & (r.tgt == r.short)].drop_duplicates(["gameId", "playId"]).copy()
    r["complete"] = r.passResult.eq("C").astype(int)
    # route depth proxy: "short" / "deep" in the play text, plus yards if complete
    r["deep"] = r.playDescription.str.contains(r"\b(?:deep)\b").astype(int)
    r["man"] = r.pff_passCoverageType.eq("Man").astype(int)
    r["third"] = (r.down >= 3).astype(int)
    r["margin_c"] = r.margin.fillna(-2).clip(-2, 2)      # never open -> worst margin
    r = r.dropna(subset=["sep_at_throw", "play_len", "yardsToGo"])
    return r


# ----------------------------------------------------------------------------- logistic regression (numpy)
def fit_logit(X, y, l2=1e-2, iters=200):
    Xb = np.c_[np.ones(len(X)), X]
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):                                   # Newton-Raphson with ridge
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) + l2 * np.r_[0, w[1:]]
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def predict(w, X):
    return 1 / (1 + np.exp(-(np.c_[np.ones(len(X)), X] @ w)))


def auc(y, p):
    order = np.argsort(p); ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    # average ranks for ties
    s = pd.Series(p).rank().values
    n1, n0 = y.sum(), len(y) - y.sum()
    return (s[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def cv_eval(df, cols, k=5, seed=11):
    rng = np.random.default_rng(seed); idx = rng.permutation(len(df)); folds = np.array_split(idx, k)
    y = df.complete.values.astype(float); X = df[cols].values.astype(float)
    mu, sd = X.mean(0), X.std(0) + 1e-9; Xs = (X - mu) / sd
    preds = np.zeros(len(df))
    for f in folds:
        tr = np.setdiff1d(idx, f)
        w = fit_logit(Xs[tr], y[tr]); preds[f] = predict(w, Xs[f])
    w_all = fit_logit(Xs, y)
    return dict(auc=auc(y, preds), logloss=logloss(y, preds), coef=dict(zip(cols, w_all[1:] )), n=len(df))


def model_section(routes):
    t = targeted(routes)
    base = ["sep_at_throw", "deep", "man", "third", "yardsToGo", "play_len"]
    full = base + ["margin_c"]
    b, f = cv_eval(t, base), cv_eval(t, full)
    sep_only = cv_eval(t, ["sep_at_throw"])
    pre = ["deep", "man", "third", "yardsToGo"]                       # known before / without the throw
    pre_b, pre_f = cv_eval(t, pre), cv_eval(t, pre + ["margin_c"])
    rows = [dict(model="pre-throw context only (depth, coverage, down/distance)", auc=pre_b["auc"], logloss=pre_b["logloss"], n=pre_b["n"]),
            dict(model="pre-throw context + race margin", auc=pre_f["auc"], logloss=pre_f["logloss"], n=pre_f["n"]),
            dict(model="separation at throw only", auc=sep_only["auc"], logloss=sep_only["logloss"], n=sep_only["n"]),
            dict(model="separation + depth + coverage + down/distance + time to throw", auc=b["auc"], logloss=b["logloss"], n=b["n"]),
            dict(model="... + race margin", auc=f["auc"], logloss=f["logloss"], n=f["n"])]
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "model_results.csv"), index=False)
    coef = pd.Series(f["coef"]).rename("standardized_coef")
    coef.to_csv(os.path.join(OUT, "model_coefficients.csv"))
    print("\nCOMPLETION MODEL (5-fold CV, pressured targeted passes, n=%d)" % f["n"])
    print(res.round(3).to_string(index=False))
    print("\nstandardized coefficients (full model):"); print(coef.round(3).to_string())
    # partial dependence: P(complete) vs margin holding others at mean
    y = t.complete.values.astype(float); X = t[full].values.astype(float); mu, sd = X.mean(0), X.std(0) + 1e-9
    w = fit_logit((X - mu) / sd, y)
    grid = np.linspace(-1.5, 2.0, 36); Xg = np.tile(mu, (len(grid), 1)); Xg[:, -1] = grid
    pg = predict(w, (Xg - mu) / sd)
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=160)
    ax.plot(grid, 100 * pg, color="#2a9d6f", lw=2.5)
    ax.axvline(0, color="#c0392b", ls="--", lw=1.2); ax.text(0.05, 100 * pg.min() + 1, "pocket breaks before he is open  |  open before pocket breaks", fontsize=8.5, color=MUTED)
    ax.set_xlabel("race margin of the targeted receiver (seconds open before the pocket broke)")
    ax.set_ylabel("modelled completion probability (%)")
    ax.set_title("Completion probability rises with the race margin, holding separation, depth and coverage fixed",
                 loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=14)
    ax.text(0, 1.02, f"Logistic regression, pressured targeted passes (n={f['n']}). Adding the margin moves 5-fold AUC {b['auc']:.3f} -> {f['auc']:.3f}.",
            transform=ax.transAxes, color=MUTED, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "chart_model.png")); plt.close(fig)
    return res


# ----------------------------------------------------------------------------- robustness
def headline(routes):
    t = targeted(routes)
    p = routes[routes.pressured]
    plays = p.groupby(["gameId", "playId"]).agg(won=("won", "max"), res=("passResult", "first"))
    lb = routes[routes.officialPosition.isin(["WR", "TE"])].groupby("displayName").agg(n=("won", "size"), p=("pressured", "sum"), win=("won", lambda s: s[routes.loc[s.index, "pressured"]].mean()))
    lb = lb[(lb.n >= 40) & (lb.p >= 10)].sort_values("win", ascending=False)
    return dict(pressured_plays=int(plays.shape[0]), pressure_rate=routes.drop_duplicates(["gameId", "playId"]).pressured.mean(),
                median_pocket=p.t_pressure.median(), median_open=routes.t_open.median(),
                comp_target_won=t[t.won].complete.mean(), comp_target_lost=t[~t.won].complete.mean(), n_targets=len(t),
                sack_nobody_won=plays[~plays.won].res.eq("S").mean(), sack_someone_won=plays[plays.won].res.eq("S").mean(),
                top3=", ".join(lb.head(3).index.str.split().str[-1]))


def robustness_section():
    files = sorted(glob.glob(os.path.join(OUT, "race_routes_*.csv")))
    if not files:
        print("\n(no sensitivity files; run e.g.  RACE_PRESSURE=1.5 python compute_race.py --games 30)"); return None
    base = pd.read_csv(os.path.join(OUT, "race_routes.csv"))
    games = sorted(base.gameId.unique())[:30]
    rows = [dict(setting="baseline 2.0 / 3.0 / 5.0 (same 30 games)", **headline(base[base.gameId.isin(games)]))]
    for f in files:
        tag = os.path.basename(f).replace("race_routes_", "").replace(".csv", "").split("_")
        rows.append(dict(setting=f"pressure {tag[0]} / open {tag[1]} / stem {tag[2]}", **headline(pd.read_csv(f))))
    rob = pd.DataFrame(rows)
    rob.to_csv(os.path.join(OUT, "robustness.csv"), index=False)
    print("\nROBUSTNESS to thresholds (30-game subset):")
    print(rob.round(3).to_string(index=False))
    return rob


if __name__ == "__main__":
    routes = pd.read_csv(os.path.join(OUT, "race_routes.csv"))
    model_section(routes)
    robustness_section()
    print("\nwrote model_results.csv, model_coefficients.csv, chart_model.png, robustness.csv")

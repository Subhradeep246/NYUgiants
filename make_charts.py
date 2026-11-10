"""
THE RACE - charts
-----------------
Reads output/race_*.csv (from compute_race.py) and writes four PNGs:
  output/chart_receivers.png  scout view  - route clock vs. win rate, one dot per receiver
  output/chart_teams.png      coach view  - protection/route sync per offense
  output/chart_positions.png  the headline - how often each position wins the race
  output/chart_value.png      scout view 2 - win rate vs EPA per target and vs NGS separation (nflverse)
Usage: python make_charts.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

POS_COLOR = {"WR": "#2f6fdb", "TE": "#e07b2a", "RB": "#2a9d6f", "FB": "#8a8f98"}   # fixed order, never cycled
INK, MUTED, GRID = "#1f2630", "#6b7480", "#e3e7ec"

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .8})


def receivers_chart(lb, routes):
    pocket_med = routes.drop_duplicates(["gameId", "playId"]).t_pressure.median()
    lb = lb[lb.officialPosition.isin(["WR", "TE", "RB"])].copy()
    fig, ax = plt.subplots(figsize=(11, 7), dpi=160)
    for pos, c in POS_COLOR.items():
        d = lb[lb.officialPosition == pos]
        if d.empty:
            continue
        ax.scatter(d.mean_t_open, 100 * d.win_rate_under_pressure, s=12 + d.routes * .9, c=c,
                   alpha=.75, edgecolor="white", linewidth=1, label=f"{pos} (n={len(d)})")
    ax.axvline(pocket_med, color="#c0392b", ls="--", lw=1.5)
    ax.text(pocket_med + .03, 2, f"league median pocket clock {pocket_med:.1f}s", color="#c0392b", fontsize=9, va="bottom")
    # label the standouts only
    big = lb[(lb.routes >= 80) & lb.officialPosition.isin(["WR", "TE"])]
    lab = pd.concat([big.nlargest(7, "win_rate_under_pressure"), big.nsmallest(3, "mean_t_open"),
                     big.nsmallest(4, "win_rate_under_pressure"),
                     lb[lb.officialPosition == "RB"].nlargest(2, "routes")]).drop_duplicates("nflId")
    for _, r in lab.iterrows():
        ax.annotate(r.displayName.split(" ")[-1], (r.mean_t_open, 100 * r.win_rate_under_pressure),
                    xytext=(6, 4), textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xlabel("Route clock - average seconds after snap until open (3 yds separation, after a 5 yd stem)")
    ax.set_ylabel("Win rate when pressured - % of pressured dropbacks open before the pocket broke")
    ax.set_title("THE RACE  ·  who gets open before the pocket breaks?", loc="left", fontsize=15, fontweight="bold", color=INK, pad=22)
    ax.text(0, 1.02, "One dot per receiver (min 30 routes, 10 under pressure), 2021 weeks 1-8. Dot size = routes run. Labels: high-volume WR/TE standouts.",
            transform=ax.transAxes, color=MUTED, fontsize=9.5)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="lower left", title="position", title_fontsize=9, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chart_receivers.png"))
    plt.close(fig)


def teams_chart(teams):
    t = teams.sort_values("median_sync")
    fig, ax = plt.subplots(figsize=(9, 8), dpi=160)
    colors = ["#2a9d6f" if v >= t.median_sync.median() else "#c0392b" for v in t.median_sync]
    bars = ax.barh(t.possessionTeam, t.median_sync, color=colors, height=.68)
    for b, v, pr in zip(bars, t.median_sync, t.pressure_rate):
        ax.text(v + .01, b.get_y() + b.get_height() / 2, f"{v:.2f}s   (pressured {100*pr:.0f}%)", va="center", fontsize=8.5, color=INK)
    ax.axvline(t.median_sync.median(), color=MUTED, lw=1, ls=":")
    ax.set_xlabel("Protection-route sync: seconds between first receiver open and pocket breaking (median per dropback)")
    ax.set_title("THE RACE  ·  which offenses out-run their own protection?", loc="left", fontsize=14, fontweight="bold", color=INK, pad=20)
    ax.text(0, 1.015, "Bigger = routes come open well before the pocket breaks. Small = QB is throwing into a collapsing pocket with nobody open.",
            transform=ax.transAxes, color=MUTED, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, t.median_sync.max() * 1.35)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chart_teams.png"))
    plt.close(fig)


def value_chart(lb):
    """Scout view 2 (nflverse): does winning the race early go with value per target, and is it just
    NGS separation under another name?  Two panels, one dot per WR/TE."""
    need = {"epa_per_target", "targets", "ngs_avg_separation"}
    if not need.issubset(lb.columns) or lb.epa_per_target.isna().all():
        print("value chart skipped (no nflverse columns in race_players.csv)"); return
    d = lb[lb.officialPosition.isin(["WR", "TE"]) & (lb.routes >= 80) & (lb.targets >= 20)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), dpi=160)
    panels = [(axes[0], "epa_per_target", "EPA per target (nflverse)", "Winning early is not the same as being valuable"),
              (axes[1], "ngs_avg_separation", "NGS average separation at the catch point (yds)", "...and it is not NGS separation either")]
    for ax, col, ylab, title in panels:
        e = d.dropna(subset=[col])
        for pos in ["WR", "TE"]:
            q = e[e.officialPosition == pos]
            ax.scatter(100 * q.win_rate_under_pressure, q[col], s=10 + q.targets * 1.6, c=POS_COLOR[pos], alpha=.75,
                       edgecolor="white", linewidth=1, label=f"{pos} (n={len(q)})")
        r = np.corrcoef(e.win_rate_under_pressure, e[col])[0, 1]
        ax.axvline(100 * e.win_rate_under_pressure.median(), color=MUTED, lw=1, ls=":")
        ax.axhline(e[col].median(), color=MUTED, lw=1, ls=":")
        lab = pd.concat([e.nlargest(4, col), e.nsmallest(2, col), e.nlargest(3, "win_rate_under_pressure"),
                         e.nsmallest(3, "win_rate_under_pressure")]).drop_duplicates("nflId")
        for _, x in lab.iterrows():
            ax.annotate(x.displayName.split(" ")[-1], (100 * x.win_rate_under_pressure, x[col]), xytext=(6, 4),
                        textcoords="offset points", fontsize=8.5, color=INK)
        ax.set_xlabel("Win rate when pressured (% of pressured routes open before the pocket broke)")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=18)
        ax.text(0, 1.015, f"correlation r = {r:+.2f}  ·  {len(e)} WR/TE with 80+ routes and 20+ targets. Dot size = targets.",
                transform=ax.transAxes, color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, loc="lower center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chart_value.png"))
    plt.close(fig)


def positions_chart(routes):
    r = routes[routes.pressured & routes.officialPosition.isin(["WR", "TE", "RB"])]
    g = r.groupby("officialPosition").agg(win=("won", "mean"), n=("won", "size"), t=("t_open", "median")).loc[["WR", "TE", "RB"]]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
    bars = ax.bar(g.index, 100 * g.win, color=[POS_COLOR[p] for p in g.index], width=.6)
    for b, (p, row) in zip(bars, g.iterrows()):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{100*row.win:.0f}%\nopen at {row.t:.1f}s", ha="center", fontsize=10, color=INK)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of pressured dropbacks open before the pocket broke")
    ax.set_title("When the pocket breaks, who has already won?", loc="left", fontsize=13, fontweight="bold", color=INK, pad=20)
    ax.text(0, 1.03, f"{r.drop_duplicates(['gameId','playId']).shape[0]:,} pressured dropbacks, 2021 weeks 1-8", transform=ax.transAxes, color=MUTED, fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chart_positions.png"))
    plt.close(fig)


if __name__ == "__main__":
    routes = pd.read_csv(os.path.join(OUT, "race_routes.csv"))
    lb = pd.read_csv(os.path.join(OUT, "race_players.csv"))
    teams = pd.read_csv(os.path.join(OUT, "race_teams.csv"))
    receivers_chart(lb, routes)
    teams_chart(teams)
    positions_chart(routes)
    value_chart(lb)
    print("wrote chart_receivers.png, chart_teams.png, chart_positions.png, chart_value.png in", OUT)

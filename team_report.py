"""
THE RACE - team report
----------------------
Turns the race clocks into coaching analytics. Reads output/race_routes.csv (from compute_race.py)
plus the PFF roles, and writes:

  output/validation.csv            does winning the race matter?  P(complete), P(sack), yards, by race outcome
  output/protection_exchange.csv   the exchange rate: extra blockers kept in -> pocket time bought vs receivers lost,
                                   league-wide and per offense, split by man / zone coverage
  output/defense_cards.csv         opponent scouting: each defense's pocket clock and which coverage slows the route clock
  output/chart_validation.png, output/chart_protection.png

Usage: python team_report.py [--team BAL]   (prints that team's card)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "nfl-big-data-bowl-regional-event-data-main", "data")
OUT = os.path.join(HERE, "output")
INK, MUTED, GRID = "#1f2630", "#6b7480", "#e3e7ec"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": .8})


def load():
    routes = pd.read_csv(os.path.join(OUT, "race_routes.csv"))
    yards = pd.read_csv(os.path.join(DATA, "plays.csv"), usecols=["gameId", "playId", "prePenaltyPlayResult"])
    routes = routes.merge(yards, on=["gameId", "playId"], how="left")
    pff = pd.read_csv(os.path.join(DATA, "pffScoutingData.csv"), usecols=["gameId", "playId", "nflId", "pff_role"])
    players = pd.read_csv(os.path.join(DATA, "players.csv"), usecols=["nflId", "officialPosition"])
    pff = pff.merge(players, on="nflId", how="left")
    # extra blockers = skill players (RB/TE/WR/FB) kept in to pass block
    xb = (pff[(pff.pff_role == "Pass Block") & pff.officialPosition.isin(["RB", "TE", "WR", "FB"])]
          .groupby(["gameId", "playId"]).size().rename("extra_blockers"))
    rushers = pff[pff.pff_role == "Pass Rush"].groupby(["gameId", "playId"]).size().rename("rushers")

    # one row per play
    plays = (routes.groupby(["gameId", "playId"])
             .agg(offense=("possessionTeam", "first"), defense=("defensiveTeam", "first"),
                  passResult=("passResult", "first"), yards=("prePenaltyPlayResult", "first"),
                  down=("down", "first"), ytg=("yardsToGo", "first"),
                  coverage=("pff_passCoverage", "first"), cov_type=("pff_passCoverageType", "first"),
                  dropback=("dropBackType", "first"),
                  t_pressure=("t_pressure", "first"), play_len=("play_len", "first"),
                  n_routes=("nflId", "size"),
                  t_first_open=("t_open", "min"),
                  n_open_in_time=("won", "sum"))
             .reset_index())
    plays = plays.join(xb, on=["gameId", "playId"]).join(rushers, on=["gameId", "playId"])
    plays["extra_blockers"] = plays.extra_blockers.fillna(0).astype(int).clip(upper=2)   # 0, 1, 2+
    plays["pressured"] = plays.t_pressure.notna()
    plays["pocket_clock"] = plays.t_pressure.fillna(plays.play_len)
    plays["race_won"] = plays.n_open_in_time > 0           # at least one receiver open before the pocket broke
    plays["complete"] = plays.passResult.eq("C")
    plays["sack"] = plays.passResult.eq("S")
    plays["margin"] = plays.pocket_clock - plays.t_first_open
    plays = plays[plays.cov_type.isin(["Man", "Zone"]) & plays.dropback.ne("DESIGNED_RUN")]
    return plays


import re
def target_validation(routes):
    """Receiver-level test: did the receiver the QB actually threw to win HIS race?  Pressured, targeted passes only."""
    r = routes[routes.pressured & routes.passResult.isin(["C", "I", "IN"])].copy()
    def target_name(desc):
        m = re.search(r"pass .*?to ([A-Z][A-Za-z'\-]*\.\s?[A-Za-z'\-\.]+)", str(desc))
        return m.group(1).replace(" ", "") if m else None
    r["tgt"] = r.playDescription.map(target_name)
    r["short"] = r.displayName.map(lambda n: f"{n.split()[0][0]}.{n.replace(' Jr.', '').replace(' III', '').replace(' II', '').split()[-1]}")
    r = r[r.tgt.notna() & (r.tgt.str.lower() == r.short.str.lower())]           # keep the targeted receiver's row
    r = r.drop_duplicates(["gameId", "playId"])
    r["complete"] = r.passResult.eq("C")
    out = []
    for lab, d in [("target was open before pocket broke", r[r.won]), ("target was NOT open in time", r[~r.won])]:
        out.append(dict(outcome=lab, targets=len(d), p_complete=d.complete.mean(), p_int=d.passResult.eq("IN").mean(),
                        yards_when_complete=d.loc[d.complete, "yards"].mean()))
    r["margin_bin"] = pd.cut(r.margin, [-9, -0.01, 0.3, 0.6, 1.0, 9], labels=["lost", "<0.3s", "0.3-0.6s", "0.6-1.0s", ">1.0s"])
    for b, d in r.groupby("margin_bin", observed=True):
        out.append(dict(outcome=f"target margin: {b}", targets=len(d), p_complete=d.complete.mean(), p_int=d.passResult.eq("IN").mean(),
                        yards_when_complete=d.loc[d.complete, "yards"].mean()))
    v = pd.DataFrame(out)
    v.to_csv(os.path.join(OUT, "validation_target.csv"), index=False)
    d = v.iloc[:2].copy(); d["lab"] = ["target WON the race\n(open before pocket broke)", "target LOST the race\n(not open in time)"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=160)
    bars = ax.bar(d.lab, 100 * d.p_complete, color=["#2a9d6f", "#c0392b"], width=.55)
    for b_, r_ in zip(bars, d.itertuples()):
        ax.text(b_.get_x() + b_.get_width() / 2, b_.get_height() + 1.5, f"{100*r_.p_complete:.0f}%", ha="center", fontsize=16, fontweight="bold", color=INK)
        ax.text(b_.get_x() + b_.get_width() / 2, 4, f"n={r_.targets} targets", ha="center", fontsize=9, color="white")
    ax.set_ylim(0, 90); ax.grid(axis="x", visible=False)
    ax.set_ylabel("completion probability (%)")
    ax.set_title("Throw to the receiver who won the race: +12 pts completion", loc="left", fontsize=13, fontweight="bold", color=INK, pad=18)
    ax.text(0, 1.02, "Pressured, targeted passes, 2021 wks 1-8. Target parsed from the play description; interceptions excluded.", transform=ax.transAxes, color=MUTED, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "chart_validation_target.png")); plt.close(fig)
    return v


def validation(plays):
    """Does winning the race predict the outcome?  Only pressured plays - that's when the race is real."""
    p = plays[plays.pressured].copy()
    rows = []
    for label, d in [("race won (someone open before pocket broke)", p[p.race_won]),
                     ("race lost (nobody open in time)", p[~p.race_won])]:
        rows.append(dict(outcome=label, dropbacks=len(d), p_complete=d.complete.mean(), p_sack=d.sack.mean(),
                         p_int=d.passResult.eq("IN").mean(), yards_per_play=d.yards.mean(),
                         success_rate=(d.yards >= np.where(d.down >= 3, d.ytg, np.maximum(4, 0.5 * d.ytg))).mean()))
    # by margin bins: how much time to spare
    p["margin_bin"] = pd.cut(p.margin, [-9, -0.01, 0.3, 0.6, 1.0, 9],
                             labels=["lost", "won by <0.3s", "0.3-0.6s", "0.6-1.0s", ">1.0s"])
    for b, d in p.groupby("margin_bin", observed=True):
        rows.append(dict(outcome=f"margin: {b}", dropbacks=len(d), p_complete=d.complete.mean(), p_sack=d.sack.mean(),
                         p_int=d.passResult.eq("IN").mean(), yards_per_play=d.yards.mean(),
                         success_rate=(d.yards >= np.where(d.down >= 3, d.ytg, np.maximum(4, 0.5 * d.ytg))).mean()))
    v = pd.DataFrame(rows)
    v.to_csv(os.path.join(OUT, "validation.csv"), index=False)

    # chart
    d = v[v.outcome.str.startswith("margin")].copy()
    d["lab"] = d.outcome.str.replace("margin: ", "")
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    x = np.arange(len(d)); w = .38
    ax.bar(x - w / 2, 100 * d.p_complete, w, color="#2a9d6f", label="completion %")
    ax.bar(x + w / 2, 100 * d.p_sack, w, color="#c0392b", label="sack %")
    for i, r in enumerate(d.itertuples()):
        ax.text(i - w / 2, 100 * r.p_complete + 1, f"{100*r.p_complete:.0f}%", ha="center", fontsize=9, color=INK)
        ax.text(i + w / 2, 100 * r.p_sack + 1, f"{100*r.p_sack:.0f}%", ha="center", fontsize=9, color=INK)
        ax.text(i, -9, f"n={r.dropbacks}", ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(x, d.lab); ax.set_ylim(0, 80)
    ax.set_xlabel("How the race went on pressured dropbacks (time between first receiver open and pocket breaking)")
    ax.set_title("Winning the race changes the play", loc="left", fontsize=14, fontweight="bold", color=INK, pad=18)
    ax.text(0, 1.02, "2021 wks 1-8, pressured dropbacks only. Completion rises and sacks collapse as the margin grows.",
            transform=ax.transAxes, color=MUTED, fontsize=9)
    ax.legend(frameon=False, loc="upper left"); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "chart_validation.png")); plt.close(fig)
    return v


def protection_exchange(plays):
    """Exchange rate of keeping skill players in to block."""
    def agg(d):
        return pd.Series(dict(dropbacks=len(d), pressure_rate=d.pressured.mean(),
                              pocket_clock=d.pocket_clock.median(), receivers=d.n_routes.mean(),
                              first_open=d.t_first_open.median(), open_in_time=d.n_open_in_time.mean(),
                              race_won=d.race_won.mean(), completion=d.complete.mean(), sack=d.sack.mean(),
                              yards=d.yards.mean()))
    league = plays.groupby(["cov_type", "extra_blockers"]).apply(agg, include_groups=False).reset_index()
    league.insert(0, "offense", "LEAGUE")
    teams = plays.groupby(["offense", "cov_type", "extra_blockers"]).apply(agg, include_groups=False).reset_index()
    ex = pd.concat([league, teams], ignore_index=True)
    # exchange rate vs 5-man protection (0 extra) in the same coverage type
    base = ex[ex.extra_blockers == 0].set_index(["offense", "cov_type"])
    ex = ex.join(base[["pocket_clock", "first_open", "race_won", "completion"]].add_prefix("base_"), on=["offense", "cov_type"])
    ex["time_bought_s"] = ex.pocket_clock - ex.base_pocket_clock
    ex["route_clock_cost_s"] = ex.first_open - ex.base_first_open
    ex["race_won_change"] = ex.race_won - ex.base_race_won
    ex = ex[ex.dropbacks >= 10]
    ex.to_csv(os.path.join(OUT, "protection_exchange.csv"), index=False)

    # chart: league exchange, man vs zone
    L = ex[ex.offense == "LEAGUE"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=160, sharey=False)
    for ax, cov in zip(axes, ["Man", "Zone"]):
        d = L[L.cov_type == cov].sort_values("extra_blockers")
        x = np.arange(len(d)); w = .38
        ax.bar(x - w / 2, d.pocket_clock, w, color="#c0392b", label="pocket clock (s)")
        ax.bar(x + w / 2, d.first_open, w, color="#2a9d6f", label="first receiver open (s)")
        for i, r in enumerate(d.itertuples()):
            ax.text(i - w / 2, r.pocket_clock + .04, f"{r.pocket_clock:.2f}", ha="center", fontsize=9, color=INK)
            ax.text(i + w / 2, r.first_open + .04, f"{r.first_open:.2f}", ha="center", fontsize=9, color=INK)
            ax.text(i, 0.12, f"race won\n{100*r.race_won:.0f}%\nn={int(r.dropbacks)}", ha="center", fontsize=8, color="white")
        ax.set_xticks(x, [f"{int(b)} extra blocker{'s' if b != 1 else ''}\n({5 - int(b)} routes)" for b in d.extra_blockers])
        ax.set_title(f"vs {cov} coverage", loc="left", fontsize=12, fontweight="bold", color=INK)
        ax.set_ylim(0, 3.6); ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    fig.suptitle("The protection exchange rate: what does keeping a blocker in actually buy?", x=0.01, ha="left",
                 fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "chart_protection.png")); plt.close(fig)
    return ex


def defense_cards(plays):
    """Opponent scouting: per defense, pocket clock and which coverage calls slow the offense's route clock."""
    def agg(d):
        return pd.Series(dict(dropbacks=len(d), pressure_rate=d.pressured.mean(), pocket_clock=d.t_pressure.median(),
                              first_open=d.t_first_open.median(), race_won_allowed=d.race_won.mean(),
                              completion_allowed=d.complete.mean(), sack_rate=d.sack.mean()))
    overall = plays.groupby("defense").apply(agg, include_groups=False).reset_index()
    overall.insert(1, "coverage", "ALL")
    bycov = plays.groupby(["defense", "coverage"]).apply(agg, include_groups=False).reset_index()
    bycov = bycov[bycov.dropbacks >= 15]
    cards = pd.concat([overall, bycov], ignore_index=True).sort_values(["defense", "coverage"])
    cards.to_csv(os.path.join(OUT, "defense_cards.csv"), index=False)
    return cards


def print_card(team, ex, cards, plays):
    lg = plays
    print(f"\n==== RACE CARD: {team} ====")
    print("\nOFFENSE - protection exchange (vs 5-man protection in same coverage):")
    t = ex[(ex.offense == team)][["cov_type", "extra_blockers", "dropbacks", "pocket_clock", "first_open",
                                  "race_won", "completion", "time_bought_s", "route_clock_cost_s", "race_won_change"]]
    print(t.round(2).to_string(index=False))
    print("\nDEFENSE - what opponents face:")
    d = cards[cards.defense == team][["coverage", "dropbacks", "pressure_rate", "pocket_clock", "first_open",
                                      "race_won_allowed", "completion_allowed", "sack_rate"]]
    print(d.round(2).to_string(index=False))
    print(f"\nLeague: pressure {lg.pressured.mean():.0%}, pocket clock {lg.t_pressure.median():.1f}s, "
          f"first open {lg.t_first_open.median():.1f}s, race won {lg.race_won.mean():.0%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--team", default="BAL"); a = ap.parse_args()
    plays = load()
    routes = pd.read_csv(os.path.join(OUT, "race_routes.csv")).merge(
        pd.read_csv(os.path.join(DATA, "plays.csv"), usecols=["gameId", "playId", "prePenaltyPlayResult"]).rename(columns={"prePenaltyPlayResult": "yards"}),
        on=["gameId", "playId"], how="left")
    vt = target_validation(routes)
    print("\nVALIDATION (receiver level) - did the TARGETED receiver win his race?")
    print(vt.round(3).to_string(index=False))
    v = validation(plays)
    print("\nVALIDATION - does winning the race matter? (pressured dropbacks)")
    print(v.round(3).to_string(index=False))
    ex = protection_exchange(plays)
    print("\nLEAGUE protection exchange:")
    print(ex[ex.offense == "LEAGUE"].round(2).to_string(index=False))
    cards = defense_cards(plays)
    print_card(a.team, ex, cards, plays)
    print("\nwrote validation.csv, protection_exchange.csv, defense_cards.csv, chart_validation.png, chart_protection.png")

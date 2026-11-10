"""
THE RACE: Pocket Clock vs. Route Clock
--------------------------------------
Every pass play is a race between two clocks:
  * pocket clock  - seconds after the snap until a pass rusher gets within
                    PRESSURE_RADIUS yards of the QB
  * route clock   - seconds after the snap until a route runner has at least
                    OPEN_SEP yards of separation from the nearest coverage defender

This script computes both clocks for every route runner on every dropback
in the 2021 Big Data Bowl tracking data and writes:
  output/race_routes.csv   one row per receiver-play (t_pressure, t_open, won)
  output/race_players.csv  per-receiver leaderboard (win rate, median clocks)
  output/race_teams.csv    per-offense "protection-route sync"

Usage:  python compute_race.py [--games N]   (N = number of games, default all)
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "nfl-big-data-bowl-regional-event-data-main", "data")
OUT = os.path.join(HERE, "output")

PRESSURE_RADIUS = 2.0   # yards: rusher this close to QB = pocket has broken
OPEN_SEP = 3.0          # yards: receiver this far from nearest defender = open
MIN_ROUTE_DIST = 5.0    # yards: receiver must have moved this far from his snap spot
                        # (pre-snap cushion is space you were GIVEN, not space you WON)
MIN_FRAME_AFTER_SNAP = 10  # clock can't be "won" before 1.0s (no QB throws earlier)
END_EVENTS = {"pass_forward", "qb_sack", "qb_strip_sack", "run", "autoevent_passforward",
              "fumble", "qb_slide", "pass_shovel", "handoff", "lateral"}
SNAP_EVENTS = {"ball_snap", "autoevent_ballsnap"}


def load_meta():
    plays = pd.read_csv(os.path.join(DATA, "plays.csv"))
    pff = pd.read_csv(os.path.join(DATA, "pffScoutingData.csv"))
    players = pd.read_csv(os.path.join(DATA, "players.csv"))
    games = pd.read_csv(os.path.join(DATA, "games.csv"))
    return plays, pff, players, games


def play_clocks(g, roles):
    """g: tracking rows of ONE play (all frames, all players + ball).
       roles: dict nflId -> pff_role for this play.
       Returns list of dicts, one per route runner."""
    ev = g.drop_duplicates("frameId").set_index("frameId")["event"]
    snap_frames = ev[ev.isin(SNAP_EVENTS)].index
    if len(snap_frames) == 0:
        return []
    f_snap = int(snap_frames.min())
    end_frames = ev[(ev.isin(END_EVENTS)) & (ev.index > f_snap)].index
    f_end = int(end_frames.min()) if len(end_frames) else int(g.frameId.max())
    if f_end - f_snap < MIN_FRAME_AFTER_SNAP + 1:
        return []

    g = g[(g.frameId >= f_snap) & (g.frameId <= f_end) & g.nflId.notna()].copy()
    g["role"] = g.nflId.map(roles)
    g = g[g.role.notna()]

    frames = np.arange(f_snap, f_end + 1)
    nF = len(frames)

    def grid(sub):
        """-> (ids, array[nF, nPlayers, 2]) with NaN for missing frames"""
        ids = sub.nflId.unique()
        arr = np.full((nF, len(ids), 2), np.nan)
        idx_f = (sub.frameId.values - f_snap).astype(int)
        idx_p = pd.Categorical(sub.nflId, categories=ids).codes
        arr[idx_f, idx_p, 0] = sub.x.values
        arr[idx_f, idx_p, 1] = sub.y.values
        return ids, arr

    qb = g[g.role == "Pass"]
    rush = g[g.role == "Pass Rush"]
    routes = g[g.role == "Pass Route"]
    cov = g[g.role == "Coverage"]
    if qb.empty or routes.empty or cov.empty:
        return []

    _, qb_xy = grid(qb)                    # nF x 1 x 2
    qb_xy = qb_xy[:, 0, :]

    # ---- pocket clock -------------------------------------------------
    t_pressure = np.nan
    if not rush.empty:
        _, rush_xy = grid(rush)            # nF x nR x 2
        d = np.linalg.norm(rush_xy - qb_xy[:, None, :], axis=2)   # nF x nR
        dmin = np.nanmin(d, axis=1)
        hit = np.where(dmin[MIN_FRAME_AFTER_SNAP:] <= PRESSURE_RADIUS)[0]
        if len(hit):
            t_pressure = (hit[0] + MIN_FRAME_AFTER_SNAP) / 10.0
    play_len = (f_end - f_snap) / 10.0

    # ---- route clock, each receiver ------------------------------------
    r_ids, r_xy = grid(routes)             # nF x nW x 2
    c_ids, c_xy = grid(cov)                # nF x nD x 2
    # separation matrix: nF x nW x nD
    sep = np.linalg.norm(r_xy[:, :, None, :] - c_xy[:, None, :, :], axis=3)
    nearest = np.nanmin(sep, axis=2)       # nF x nW
    nearest_id = np.nanargmin(np.nan_to_num(sep, nan=1e9), axis=2)  # nF x nW

    # distance each receiver has travelled from his snap position: nF x nW
    travelled = np.linalg.norm(r_xy - r_xy[0][None, :, :], axis=2)

    out = []
    for j, rid in enumerate(r_ids):
        s = nearest[MIN_FRAME_AFTER_SNAP:, j]
        moved = travelled[MIN_FRAME_AFTER_SNAP:, j]
        open_frames = np.where((s >= OPEN_SEP) & (moved >= MIN_ROUTE_DIST))[0]
        t_open = (open_frames[0] + MIN_FRAME_AFTER_SNAP) / 10.0 if len(open_frames) else np.nan
        sep_at_end = nearest[-1, j]
        # defender closest at the END of the window (throw) = inferred matchup
        matchup = c_ids[nearest_id[-1, j]]
        out.append(dict(nflId=int(rid), t_pressure=t_pressure, t_open=t_open,
                        play_len=play_len, sep_at_throw=sep_at_end,
                        max_sep=np.nanmax(nearest[:, j]), matchup_nflId=int(matchup)))
    return out


def main(n_games=None):
    plays, pff, players, games = load_meta()
    roles = pff[["gameId", "playId", "nflId", "pff_role"]]
    files = sorted(glob.glob(os.path.join(DATA, "tracking", "tracking_*.csv")))
    if n_games:
        files = files[:n_games]

    cache = os.path.join(OUT, "cache")
    os.makedirs(cache, exist_ok=True)
    parts = []
    for k, f in enumerate(files, 1):
        gid = int(os.path.basename(f).split("_")[1].split(".")[0])
        cf = os.path.join(cache, f"{gid}.csv")
        if os.path.exists(cf):                       # resumable: skip games already computed
            parts.append(pd.read_csv(cf))
            continue
        tr = pd.read_csv(f, usecols=["gameId", "playId", "nflId", "frameId", "x", "y", "event"])
        g_roles = roles[roles.gameId == gid]
        rows = []
        for pid, g in tr.groupby("playId", sort=False):
            rmap = g_roles[g_roles.playId == pid].set_index("nflId")["pff_role"].to_dict()
            for r in play_clocks(g, rmap):
                r.update(gameId=gid, playId=int(pid))
                rows.append(r)
        part = pd.DataFrame(rows)
        part.to_csv(cf, index=False)
        parts.append(part)
        print(f"[{k}/{len(files)}] game {gid}: {len(part)} receiver-plays", flush=True)

    df = pd.concat(parts, ignore_index=True)
    df = df.merge(plays[["gameId", "playId", "possessionTeam", "defensiveTeam", "passResult",
                         "down", "yardsToGo", "pff_passCoverage", "pff_passCoverageType",
                         "dropBackType", "pff_playAction", "playDescription"]],
                  on=["gameId", "playId"], how="left")
    df = df.merge(players[["nflId", "displayName", "officialPosition"]], on="nflId", how="left")
    df = df.merge(players[["nflId", "displayName"]].rename(
        columns={"nflId": "matchup_nflId", "displayName": "matchup_name"}),
        on="matchup_nflId", how="left")

    # ---- the race outcome ---------------------------------------------
    # pressured play: pocket broke before the ball was out
    df["pressured"] = df.t_pressure.notna()
    # receiver "won" if he was open before pressure (or at all, on unpressured plays)
    df["won"] = np.where(df.pressured, df.t_open < df.t_pressure, df.t_open.notna())
    df["margin"] = df.t_pressure.fillna(df.play_len) - df.t_open   # + = open with time to spare

    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "race_routes.csv"), index=False)

    # ---- per-receiver leaderboard ---------------------------------------
    skill = df[df.officialPosition.isin(["WR", "TE", "RB", "FB"])]
    lb = (skill.groupby(["nflId", "displayName", "officialPosition", "possessionTeam"])
          .agg(routes=("playId", "size"),
               pressured_routes=("pressured", "sum"),
               win_rate=("won", "mean"),
               win_rate_under_pressure=("won", lambda s: s[skill.loc[s.index, "pressured"]].mean()),
               median_t_open=("t_open", "median"),
               mean_t_open=("t_open", "mean"),
               open_rate=("t_open", lambda s: s.notna().mean()),
               median_margin=("margin", "median"),
               avg_sep_at_throw=("sep_at_throw", "mean"))
          .reset_index())
    lb = lb[(lb.routes >= 30) & (lb.pressured_routes >= 10)].sort_values("win_rate_under_pressure", ascending=False)
    lb.to_csv(os.path.join(OUT, "race_players.csv"), index=False)

    # ---- per-team sync --------------------------------------------------
    per_play = (df.groupby(["gameId", "playId", "possessionTeam"])
                .agg(t_pressure=("t_pressure", "first"),
                     play_len=("play_len", "first"),
                     t_first_open=("t_open", "min"))
                .reset_index())
    per_play["pocket_clock"] = per_play.t_pressure.fillna(per_play.play_len)
    per_play["sync"] = per_play.pocket_clock - per_play.t_first_open
    teams = (per_play.groupby("possessionTeam")
             .agg(dropbacks=("playId", "size"),
                  pressure_rate=("t_pressure", lambda s: s.notna().mean()),
                  median_pocket_clock=("t_pressure", "median"),
                  median_first_open=("t_first_open", "median"),
                  median_sync=("sync", "median"),
                  race_won_rate=("sync", lambda s: (s > 0).mean()))
             .reset_index().sort_values("median_sync", ascending=False))
    teams.to_csv(os.path.join(OUT, "race_teams.csv"), index=False)

    print("\nTop receivers (win rate when pocket breaks, min 30 routes):")
    print(lb.head(15).to_string(index=False))
    print("\nTeam sync (pocket clock - first receiver open, seconds):")
    print(teams.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=None)
    a = ap.parse_args()
    main(a.games)

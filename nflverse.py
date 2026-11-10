"""
THE RACE - nflverse join
------------------------
Free public data from nflverse (https://github.com/nflverse/nflverse-data), allowed by the competition rules.
Big Data Bowl gameId/playId match nflverse old_game_id/play_id, and Big Data Bowl nflId matches the
nfl_id column of the nflverse players file, so everything joins on IDs, with no name matching.

  play_values()   one row per Big Data Bowl play: epa, wpa, success, air_yards, yac, cp, cpoe,
                  and target_nflId (the receiver the pass was thrown to)
  ngs_receivers() Next Gen Stats receiving, 2021 weeks 1-8, aggregated per receiver (nflId):
                  NGS avg separation / cushion at the catch point, weighted by targets

Files are downloaded once into output/cache/nflverse/ (gitignored). Every caller treats nflverse as
optional: if the download fails (offline), these functions return None and the pipeline runs as before.

Usage: python nflverse.py      # download + print a join report
"""
import functools
import os
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "output", "cache", "nflverse")
BASE = "https://github.com/nflverse/nflverse-data/releases/download"
FILES = {"pbp": "pbp/play_by_play_2021.parquet",
         "players": "players/players.parquet",
         "ngs": "nextgen_stats/ngs_receiving.parquet"}
PBP_COLS = ["old_game_id", "play_id", "epa", "wpa", "wp", "success", "air_yards", "yards_after_catch",
            "cp", "cpoe", "receiver_player_id", "pass_location"]
WEEKS = range(1, 9)          # the tracking data covers 2021 regular-season weeks 1-8
VALUE_COLS = ["epa", "wpa", "wp", "success", "air_yards", "yards_after_catch", "cp", "cpoe",
              "pass_location", "target_nflId"]


def _path(key):
    """Local copy of an nflverse file, downloading it on first use. None if unavailable."""
    p = os.path.join(CACHE, os.path.basename(FILES[key]))
    if not os.path.exists(p):
        os.makedirs(CACHE, exist_ok=True)
        try:
            print(f"downloading nflverse {FILES[key]} ...", flush=True)
            urllib.request.urlretrieve(f"{BASE}/{FILES[key]}", p + ".part")
            os.replace(p + ".part", p)
        except Exception as e:                       # offline, rate-limited, moved file
            print(f"  nflverse unavailable ({e}); continuing without it")
            if os.path.exists(p + ".part"):
                os.remove(p + ".part")
            return None
    return p


def _gsis_to_nflid():
    p = _path("players")
    if p is None:
        return None
    pl = pd.read_parquet(p, columns=["gsis_id", "nfl_id"]).dropna()
    pl["nfl_id"] = pd.to_numeric(pl.nfl_id, errors="coerce")
    return pl.dropna().drop_duplicates("gsis_id").set_index("gsis_id").nfl_id.astype(int)


@functools.lru_cache(maxsize=1)
def play_values():
    """gameId, playId -> epa, wpa, wp, success, air_yards, yards_after_catch, cp, cpoe, pass_location, target_nflId."""
    p = _path("pbp")
    ids = _gsis_to_nflid()
    if p is None or ids is None:
        return None
    pbp = pd.read_parquet(p, columns=PBP_COLS).dropna(subset=["old_game_id", "play_id"])
    pbp["gameId"] = pbp.old_game_id.astype("int64")
    pbp["playId"] = pbp.play_id.astype("int64")
    pbp["target_nflId"] = pbp.receiver_player_id.map(ids)
    return pbp.drop(columns=["old_game_id", "play_id", "receiver_player_id"]).drop_duplicates(["gameId", "playId"])


def ngs_receivers():
    """nflId -> ngs_targets, ngs_avg_separation, ngs_avg_cushion, ngs_avg_intended_air_yards (2021 wks 1-8)."""
    p = _path("ngs")
    ids = _gsis_to_nflid()
    if p is None or ids is None:
        return None
    n = pd.read_parquet(p)
    n = n[(n.season == 2021) & (n.season_type == "REG") & n.week.isin(WEEKS)].copy()
    n["nflId"] = n.player_gsis_id.map(ids)
    n = n.dropna(subset=["nflId"])
    w = n.targets.clip(lower=1)
    agg = (n.assign(sep_w=n.avg_separation * w, cush_w=n.avg_cushion * w, iay_w=n.avg_intended_air_yards * w, w=w)
           .groupby("nflId")[["sep_w", "cush_w", "iay_w", "w", "targets"]].sum())
    out = pd.DataFrame({"ngs_targets": agg.targets,
                        "ngs_avg_separation": agg.sep_w / agg.w,
                        "ngs_avg_cushion": agg.cush_w / agg.w,
                        "ngs_avg_intended_air_yards": agg.iay_w / agg.w})
    out.index = out.index.astype(int)
    return out.reset_index()


def play_value(game, play):
    """What one play was worth, for the broadcast mock: dict(epa, wpa, air_yards) or {} if unavailable."""
    v = play_values()
    if v is None:
        return {}
    row = v[(v.gameId == game) & (v.playId == play)]
    if row.empty:
        return {}
    r = row.iloc[0]
    clean = lambda x: None if pd.isna(x) else round(float(x), 3)
    return dict(epa=clean(r.epa), wpa=clean(r.wpa), air_yards=clean(r.air_yards),
                target_nflId=None if pd.isna(r.target_nflId) else int(r.target_nflId))


def add_values(df):
    """Join nflverse play values onto a frame with gameId/playId/nflId/playDescription and add `targeted`
    (this row is the receiver the ball was thrown to). Uses nflverse IDs; falls back to parsing the
    play description if nflverse is unavailable."""
    v = play_values()
    if v is not None:
        df = df.merge(v, on=["gameId", "playId"], how="left")   # merge copies, so the cached frame is untouched
        df["targeted"] = df.nflId.eq(df.target_nflId)
        return df
    import re
    def tgt(d):
        m = re.search(r"pass .*?to ([A-Z][A-Za-z'\-]*\.\s?[A-Za-z'\-\.]+)", str(d))
        # rstrip: "pass incomplete short left to A.Cooper." would otherwise keep the full stop
        return m.group(1).replace(" ", "").rstrip(".").lower() if m else None
    short = df.displayName.fillna("?").map(
        lambda n: f"{n.split()[0][0]}.{n.replace(' Jr.', '').replace(' III', '').replace(' II', '').split()[-1]}".lower())
    df["targeted"] = df.playDescription.map(tgt).eq(short)
    return df


if __name__ == "__main__":
    data = os.path.join(HERE, "nfl-big-data-bowl-regional-event-data-main", "data")
    plays = pd.read_csv(os.path.join(data, "plays.csv"), usecols=["gameId", "playId", "passResult"])
    v = play_values()
    if v is None:
        raise SystemExit("nflverse not available")
    m = plays.merge(v, on=["gameId", "playId"], how="left")
    att = m.passResult.isin(["C", "I", "IN"])
    print(f"plays matched to nflverse: {m.epa.notna().sum():,} / {len(m):,}")
    print(f"pass attempts with a target nflId: {m.loc[att, 'target_nflId'].notna().sum():,} / {att.sum():,}")
    print("mean EPA by passResult:", m.groupby("passResult").epa.mean().round(2).to_dict())
    n = ngs_receivers()
    print(f"NGS receivers (wks 1-8, matched to nflId): {len(n) if n is not None else 0}")

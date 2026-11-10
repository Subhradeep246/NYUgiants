"""
THE RACE - broadcast mock
-------------------------
Builds a self-contained HTML animation of one play, TV-overlay style:
a top-down field with all 22 players, and two live clocks racing at the top:
  POCKET CLOCK  (red)   - fills as the nearest rusher closes on the QB
  ROUTE CLOCK   (green) - one lane per receiver, fills as he creates separation
The moment a clock "wins" its lane flashes with the time.

Usage:
  python race_tv.py                      # auto-picks a dramatic play from output/race_routes.csv
  python race_tv.py --game 2021090900 --play 97
Output: output/race_tv_<game>_<play>.html  (open in any browser; no internet needed)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "nfl-big-data-bowl-regional-event-data-main", "data")
OUT = os.path.join(HERE, "output")

from compute_race import PRESSURE_RADIUS, OPEN_SEP, MIN_ROUTE_DIST, MIN_FRAME_AFTER_SNAP, \
    END_EVENTS, SNAP_EVENTS


def pick_play(routes):
    """A dramatic play: pressure arrived, at least one receiver won the race by < 0.6s,
    and the ball went somewhere (complete or incomplete) - on a game we have tracking for."""
    have = {int(f.split("_")[1].split(".")[0]) for f in os.listdir(os.path.join(DATA, "tracking"))}
    r = routes[routes.gameId.isin(have) & routes.pressured & routes.won &
               routes.passResult.isin(["C", "I"]) & (routes.margin.between(0.1, 0.6))]
    r = r[r.officialPosition.isin(["WR", "TE"])]
    if r.empty:
        r = routes[routes.gameId.isin(have) & routes.pressured]
    row = r.sort_values("t_pressure").iloc[len(r) // 2]
    return int(row.gameId), int(row.playId)


def build(game, play):
    plays = pd.read_csv(os.path.join(DATA, "plays.csv"))
    pff = pd.read_csv(os.path.join(DATA, "pffScoutingData.csv"))
    players = pd.read_csv(os.path.join(DATA, "players.csv"))
    games = pd.read_csv(os.path.join(DATA, "games.csv"))
    tr = pd.read_csv(os.path.join(DATA, "tracking", f"tracking_{game}.csv"))
    g = tr[tr.playId == play].copy()
    meta = plays[(plays.gameId == game) & (plays.playId == play)].iloc[0]
    gm = games[games.gameId == game].iloc[0]
    roles = pff[(pff.gameId == game) & (pff.playId == play)].set_index("nflId")
    names = players.set_index("nflId")

    # normalise so offense always goes left -> right
    if g.playDirection.iloc[0] == "left":
        g["x"] = 120 - g.x
        g["y"] = 53.3 - g.y

    ev = g.drop_duplicates("frameId").set_index("frameId")["event"]
    f_snap = int(ev[ev.isin(SNAP_EVENTS)].index.min())
    ends = ev[(ev.isin(END_EVENTS)) & (ev.index > f_snap)].index
    f_end = int(ends.min()) if len(ends) else int(g.frameId.max())
    f0 = max(1, f_snap - 10)                      # show 1s pre-snap
    f1 = min(int(g.frameId.max()), f_end + 8)     # and a beat after the throw
    g = g[(g.frameId >= f0) & (g.frameId <= f1)]

    frames = sorted(g.frameId.unique())
    ids = [i for i in g.nflId.dropna().unique()]
    role = {int(i): roles.pff_role.get(i, "") for i in ids}
    qb = [i for i in ids if role[int(i)] == "Pass"]
    rushers = [i for i in ids if role[int(i)] == "Pass Rush"]
    receivers = [i for i in ids if role[int(i)] == "Pass Route"]
    defenders = [i for i in ids if role[int(i)] == "Coverage"]

    pos = {int(i): g[g.nflId == i].set_index("frameId")[["x", "y"]].reindex(frames).values.tolist()
           for i in ids}
    ball = g[g.nflId.isna()].set_index("frameId")[["x", "y"]].reindex(frames).values.tolist()

    def xy(i, k):
        return np.array(pos[int(i)][k])

    # per-frame metrics
    pocket = []      # nearest rusher distance to QB
    sep = {int(r): [] for r in receivers}
    moved = {int(r): [] for r in receivers}
    for k, f in enumerate(frames):
        q = xy(qb[0], k)
        pocket.append(float(min(np.linalg.norm(xy(r, k) - q) for r in rushers)) if rushers else None)
        for r in receivers:
            p = xy(r, k)
            d = min(np.linalg.norm(xy(c, k) - p) for c in defenders) if defenders else 99
            sep[int(r)].append(float(d))
            snap_k = frames.index(f_snap)
            moved[int(r)].append(float(np.linalg.norm(p - xy(r, snap_k))))

    offense = meta.possessionTeam
    desc = meta.playDescription
    payload = dict(
        game=int(game), play=int(play), frames=[int(f) for f in frames], f_snap=f_snap, f_end=f_end,
        title=f"{gm.visitorTeamAbbr} @ {gm.homeTeamAbbr}  ·  Week {int(gm.week)}, 2021  ·  Q{int(meta.quarter)} {meta.gameClock}",
        situation=f"{int(meta.down)}{'st' if meta.down == 1 else 'nd' if meta.down == 2 else 'rd' if meta.down == 3 else 'th'} & {int(meta.yardsToGo)}  ·  {meta.pff_passCoverage} ({meta.pff_passCoverageType})",
        desc=desc, offense=offense, defense=meta.defensiveTeam,
        los=float(meta.absoluteYardlineNumber) if g.playDirection.iloc[0] == "right" else 120 - float(meta.absoluteYardlineNumber),
        players=[dict(id=int(i), name=names.displayName.get(i, "?"), pos=names.officialPosition.get(i, ""),
                      team=str(g[g.nflId == i].team.iloc[0]), role=role[int(i)], xy=pos[int(i)],
                      jersey=int(g[g.nflId == i].jerseyNumber.iloc[0])) for i in ids],
        ball=ball, pocket=pocket,
        receivers=[dict(id=int(r), name=names.displayName.get(r, "?"), pos=names.officialPosition.get(r, ""),
                        sep=sep[int(r)], moved=moved[int(r)]) for r in receivers],
        thresholds=dict(pressure=PRESSURE_RADIUS, open=OPEN_SEP, route=MIN_ROUTE_DIST,
                        min_frames=MIN_FRAME_AFTER_SNAP),
    )
    return payload


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>THE RACE · Pocket Clock vs Route Clock</title>
<style>
:root{--bg:#07090d;--panel:rgba(16,22,30,.86);--ink:#f3f5f7;--muted:#9aa5b1;--red:#ff3b3b;--green:#2ee06f;--gold:#ffd166;--line:#2a3340}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(255,59,59,.7)}100%{box-shadow:0 0 0 14px rgba(255,59,59,0)}}
@keyframes pulseg{0%{box-shadow:0 0 0 0 rgba(46,224,111,.7)}100%{box-shadow:0 0 0 14px rgba(46,224,111,0)}}
@keyframes pop{0%{transform:scale(1)}40%{transform:scale(1.18)}100%{transform:scale(1)}}
.mode{display:flex;gap:6px}.mode button.on{border-color:var(--gold);color:var(--gold)}
body.lt .clocks,body.lt .top,body.lt .foot{display:none}
body.lt canvas{opacity:.35}
.l3{display:none;position:relative;margin-top:-150px;z-index:2;padding:0 18px}
body.lt .l3{display:block}
.l3 .box{background:linear-gradient(90deg,rgba(7,9,13,.96),rgba(7,9,13,.86));border:1px solid var(--line);border-left:6px solid var(--gold);border-radius:8px;padding:12px 18px;display:grid;grid-template-columns:230px 1fr 1fr;gap:18px;align-items:center;backdrop-filter:blur(6px)}
.l3 .tag{font-weight:900;letter-spacing:.16em;font-size:12px;color:var(--gold)}
.l3 .big{font-size:40px;font-weight:900;font-variant-numeric:tabular-nums;line-height:1}
.l3 .lab{font-size:12px;color:var(--muted);letter-spacing:.08em;text-transform:uppercase}
.l3 .race{display:flex;flex-direction:column;gap:6px}
.l3 .row{display:grid;grid-template-columns:150px 1fr 90px;gap:10px;align-items:center;font-size:14px;font-weight:700}
.l3 .row .bar{height:10px}.l3 .row.pocket .bar>i{background:var(--red)}.l3 .row.route .bar>i{background:var(--green)}
.l3 .row .t{text-align:right;font-variant-numeric:tabular-nums;color:var(--muted)}
.l3 .row.won .t{color:var(--green);animation:pop .4s ease-out}.l3 .row.pocket.won .t{color:var(--red)}
.l3 .row.pocket.won .bar{animation:pulse .9s ease-out infinite}.l3 .row.route.won .bar{animation:pulseg .9s ease-out infinite}
.l3 .verdict{font-size:22px;font-weight:900;color:var(--gold);min-height:28px}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,system-ui,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:16px}
.top{display:flex;justify-content:space-between;align-items:baseline;gap:16px;flex-wrap:wrap}
.brand{font-weight:900;letter-spacing:.14em;font-size:13px;color:var(--gold)}
h1{font-size:22px;margin:4px 0 0;font-weight:800}
.sub{color:var(--muted);font-size:13px}
.clocks{display:grid;grid-template-columns:1fr;gap:6px;margin:14px 0;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}
.lane{display:grid;grid-template-columns:190px 1fr 92px;align-items:center;gap:12px;font-size:14px}
.lane .name{font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.lane .name small{color:var(--muted);font-weight:500;margin-left:6px}
.bar{height:16px;background:#1b2330;border-radius:4px;position:relative;overflow:hidden}
.bar>i{position:absolute;left:0;top:0;bottom:0;width:0;border-radius:4px;transition:width .08s linear}
.lane.pocket .bar>i{background:var(--red)}.lane.route .bar>i{background:var(--green)}
.lane .t{font-variant-numeric:tabular-nums;text-align:right;font-weight:800;color:var(--muted)}
.lane.won .t{color:var(--green)}.lane.pocket.won .t{color:var(--red)}
.lane.won .bar{animation:pulseg .9s ease-out infinite}.lane.pocket.won .bar{animation:pulse .9s ease-out infinite}
.lane.won .t{animation:pop .4s ease-out}
.tick{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--gold);opacity:.9}
canvas{width:100%;height:auto;display:block;border-radius:10px;border:1px solid var(--line);background:#14532d}
.ctrl{display:flex;gap:10px;align-items:center;margin-top:10px;color:var(--muted);font-size:13px}
button{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:8px 14px;font-weight:700;cursor:pointer}
button:hover{border-color:var(--gold)}
input[type=range]{flex:1}
.banner{margin-top:10px;min-height:28px;font-weight:800;font-size:18px;color:var(--gold)}
.foot{color:var(--muted);font-size:12px;margin-top:10px;line-height:1.5}
</style></head><body><div class="wrap">
<div class="top"><div><div class="brand">THE RACE · NEXT GEN TRACKING</div><h1 id="title"></h1><div class="sub" id="sit"></div></div>
<div class="sub" id="desc" style="max-width:520px;text-align:right"></div></div>
<div class="clocks" id="clocks"></div>
<canvas id="field" width="1248" height="560"></canvas>
<div class="l3"><div class="box">
  <div><div class="tag">THE RACE</div><div class="big" id="l3clock">0.0<span style="font-size:18px;color:var(--muted)"> s</span></div><div class="lab" id="l3sit"></div></div>
  <div class="race" id="l3rows"></div>
  <div><div class="lab">verdict</div><div class="verdict" id="l3verdict">&nbsp;</div></div>
</div></div>
<div class="ctrl"><button id="play">▶ Play</button><button id="reset">↺</button><span class="mode"><button id="mAn" class="on">Analyst</button><button id="mLt">Lower-third</button></span><input type="range" id="scrub" min="0" value="0" step="1"><span id="clock">0.0 s</span></div>
<div class="banner" id="banner"></div>
<div class="foot">Pocket clock: time from snap until a pass rusher is within <b id="thP"></b> yds of the QB. Route clock: time until a receiver has <b id="thO"></b> yds of separation from the nearest coverage defender after travelling <b id="thR"></b> yds (pre-snap cushion doesn't count). Gold tick = the throw. Receivers: green · QB: gold · rushers: red · coverage: white · blockers: grey.</div>
</div>
<script>
const D = __DATA__;
const F = D.frames, n = F.length, snapK = F.indexOf(D.f_snap), endK = F.indexOf(D.f_end);
const t = k => Math.max(0, (k - snapK) / 10);
document.getElementById('title').textContent = D.title;
document.getElementById('sit').textContent = D.situation;
document.getElementById('desc').textContent = D.desc;
document.getElementById('thP').textContent = D.thresholds.pressure;
document.getElementById('thO').textContent = D.thresholds.open;
document.getElementById('thR').textContent = D.thresholds.route;

// --- compute "win" times once
const minK = snapK + D.thresholds.min_frames;
let tPressure = null;
for (let k = minK; k <= endK; k++) if (D.pocket[k] != null && D.pocket[k] <= D.thresholds.pressure) { tPressure = t(k); break; }
D.receivers.forEach(r => { r.tOpen = null; for (let k = minK; k <= endK; k++) if (r.sep[k] >= D.thresholds.open && r.moved[k] >= D.thresholds.route) { r.tOpen = t(k); break; } });
const tEnd = t(endK), maxT = Math.max(3.5, tEnd + 0.3);

// --- clock lanes
const C = document.getElementById('clocks');
function lane(cls, label, sub) { const el = document.createElement('div'); el.className = 'lane ' + cls;
  el.innerHTML = `<div class="name">${label}<small>${sub}</small></div><div class="bar"><i></i><span class="tick" style="left:${100*tEnd/maxT}%"></span></div><div class="t">—</div>`; C.appendChild(el); return el; }
const pocketLane = lane('pocket', 'POCKET CLOCK', D.defense + ' pass rush');
D.receivers.sort((a,b)=> (a.tOpen??99)-(b.tOpen??99));
D.receivers.forEach(r => r.el = lane('route', r.name, r.pos));

// --- lower-third rows
document.getElementById('l3sit').textContent = D.situation;
const L = document.getElementById('l3rows');
function l3row(cls, label) { const el = document.createElement('div'); el.className = 'row ' + cls; el.innerHTML = `<div>${label}</div><div class="bar"><i></i><span class="tick" style="left:${100*tEnd/maxT}%"></span></div><div class="t">—</div>`; L.appendChild(el); return el; }
const l3pocket = l3row('pocket', 'POCKET');
D.receivers.slice(0, 3).forEach(r => r.el3 = l3row('route', r.name.split(' ').slice(-1)[0].toUpperCase()));
document.getElementById('mAn').onclick = () => { document.body.classList.remove('lt'); document.getElementById('mAn').classList.add('on'); document.getElementById('mLt').classList.remove('on'); };
document.getElementById('mLt').onclick = () => { document.body.classList.add('lt'); document.getElementById('mLt').classList.add('on'); document.getElementById('mAn').classList.remove('on'); };

// --- field
const cv = document.getElementById('field'), ctx = cv.getContext('2d');
const X = x => x / 120 * cv.width, Y = y => (53.3 - y) / 53.3 * cv.height;
const color = p => p.role === 'Pass' ? '#ffd166' : p.role === 'Pass Route' ? '#2ee06f' : p.role === 'Pass Rush' ? '#ff3b3b' : p.role === 'Coverage' ? '#ffffff' : '#9aa5b1';
function drawField() {
  ctx.fillStyle = '#14532d'; ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.fillStyle = '#0f3d22'; ctx.fillRect(0, 0, X(10), cv.height); ctx.fillRect(X(110), 0, X(10), cv.height);
  ctx.strokeStyle = 'rgba(255,255,255,.35)'; ctx.lineWidth = 1;
  for (let x = 10; x <= 110; x += 5) { ctx.beginPath(); ctx.moveTo(X(x), 0); ctx.lineTo(X(x), cv.height); ctx.stroke(); }
  ctx.fillStyle = 'rgba(255,255,255,.6)'; ctx.font = 'bold 14px sans-serif'; ctx.textAlign = 'center';
  for (let x = 20; x <= 100; x += 10) { const yd = x <= 60 ? x - 10 : 110 - x; ctx.fillText(yd, X(x), 20); ctx.fillText(yd, X(x), cv.height - 8); }
  ctx.strokeStyle = '#4da3ff'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(X(D.los), 0); ctx.lineTo(X(D.los), cv.height); ctx.stroke();
}
function draw(k) {
  drawField();
  const q = D.players.find(p => p.role === 'Pass');
  // pressure radius ring around QB
  if (q && q.xy[k][0] != null) { ctx.beginPath(); ctx.arc(X(q.xy[k][0]), Y(q.xy[k][1]), X(D.thresholds.pressure), 0, 7); ctx.strokeStyle = 'rgba(255,59,59,.6)'; ctx.setLineDash([4,4]); ctx.stroke(); ctx.setLineDash([]); }
  // receiver -> nearest defender separation lines
  D.receivers.forEach(r => { const p = D.players.find(p => p.id === r.id); if (!p || p.xy[k][0] == null) return;
    let best = null, bd = 1e9; D.players.filter(d => d.role === 'Coverage').forEach(d => { if (d.xy[k][0] == null) return; const dd = Math.hypot(d.xy[k][0]-p.xy[k][0], d.xy[k][1]-p.xy[k][1]); if (dd < bd) { bd = dd; best = d; } });
    if (best) { ctx.beginPath(); ctx.moveTo(X(p.xy[k][0]), Y(p.xy[k][1])); ctx.lineTo(X(best.xy[k][0]), Y(best.xy[k][1])); ctx.strokeStyle = bd >= D.thresholds.open ? 'rgba(46,224,111,.8)' : 'rgba(255,255,255,.35)'; ctx.lineWidth = 2; ctx.stroke(); }
  });
  // players
  D.players.forEach(p => { const [x, y] = p.xy[k]; if (x == null) return;
    ctx.beginPath(); ctx.arc(X(x), Y(y), 9, 0, 7); ctx.fillStyle = color(p); ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = '#0b0f14'; ctx.stroke();
    ctx.fillStyle = '#0b0f14'; ctx.font = 'bold 10px sans-serif'; ctx.textAlign = 'center'; ctx.fillText(p.jersey, X(x), Y(y) + 4);
    if (p.role === 'Pass Route') { ctx.fillStyle = '#fff'; ctx.font = 'bold 12px sans-serif'; ctx.fillText(p.name.split(' ').slice(-1)[0], X(x), Y(y) - 14); }
  });
  const b = D.ball[k]; if (b && b[0] != null) { ctx.beginPath(); ctx.ellipse(X(b[0]), Y(b[1]), 6, 4, 0, 0, 7); ctx.fillStyle = '#8b4513'; ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.stroke(); }
  // clocks
  const now = t(k);
  const setLane = (el, prog, label, won, wonAt) => { el.querySelector('i').style.width = (100 * Math.min(prog, maxT) / maxT) + '%'; el.querySelector('.t').textContent = label; el.classList.toggle('won', won); };
  const pw = tPressure != null && now >= tPressure;
  setLane(pocketLane, k < snapK ? 0 : (pw ? tPressure : now), k < snapK ? 'pre-snap' : pw ? `BROKE ${tPressure.toFixed(1)}s` : `${D.pocket[k] != null ? D.pocket[k].toFixed(1) + ' yds' : ''}`, pw);
  D.receivers.forEach(r => { const w = r.tOpen != null && now >= r.tOpen; setLane(r.el, k < snapK ? 0 : (w ? r.tOpen : now), k < snapK ? 'pre-snap' : w ? `OPEN ${r.tOpen.toFixed(1)}s` : `${r.sep[k].toFixed(1)} yds`, w); });
  document.getElementById('clock').textContent = (k < snapK ? '-' : '') + Math.abs((k - snapK) / 10).toFixed(1) + ' s';
  document.getElementById('l3clock').firstChild.textContent = (k < snapK ? '-' : '') + Math.abs((k - snapK) / 10).toFixed(1);
  setLane(l3pocket, k < snapK ? 0 : (pw ? tPressure : now), k < snapK ? '—' : pw ? `${tPressure.toFixed(1)}s` : `${D.pocket[k] != null ? D.pocket[k].toFixed(1) + ' yd' : ''}`, pw);
  D.receivers.slice(0, 3).forEach(r => { const w = r.tOpen != null && now >= r.tOpen; setLane(r.el3, k < snapK ? 0 : (w ? r.tOpen : now), k < snapK ? '—' : w ? `${r.tOpen.toFixed(1)}s` : `${r.sep[k].toFixed(1)} yd`, w); });
  const bn = document.getElementById('banner');
  if (k < snapK) bn.textContent = '';
  else if (k >= endK) { const winners = D.receivers.filter(r => r.tOpen != null && (tPressure == null || r.tOpen < tPressure));
    bn.textContent = tPressure == null ? `Clean pocket for ${tEnd.toFixed(1)}s — ${winners.length} receiver(s) open` : winners.length ? `${winners.map(r=>r.name.split(' ').slice(-1)[0]).join(', ')} won the race — open ${(tPressure - winners[0].tOpen).toFixed(1)}s before the pocket broke` : `Pocket broke at ${tPressure.toFixed(1)}s — nobody was open in time`; }
  else bn.textContent = pw ? '🔴 POCKET BROKEN' : '';
  document.getElementById('l3verdict').innerHTML = bn.textContent || '&nbsp;';
}
let k = 0, timer = null; const scrub = document.getElementById('scrub'); scrub.max = n - 1;
function step() { k++; if (k >= n) { k = n - 1; stop(); } scrub.value = k; draw(k); }
function play() { if (timer) return stop(); if (k >= n - 1) k = 0; timer = setInterval(step, 100); document.getElementById('play').textContent = '❚❚ Pause'; }
function stop() { clearInterval(timer); timer = null; document.getElementById('play').textContent = '▶ Play'; }
document.getElementById('play').onclick = play; document.getElementById('reset').onclick = () => { stop(); k = 0; scrub.value = 0; draw(0); };
scrub.oninput = e => { stop(); k = +e.target.value; draw(k); };
draw(0);
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int)
    ap.add_argument("--play", type=int)
    a = ap.parse_args()
    if a.game is None:
        routes = pd.read_csv(os.path.join(OUT, "race_routes.csv"))
        a.game, a.play = pick_play(routes)
    payload = build(a.game, a.play)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"race_tv_{a.game}_{a.play}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(HTML.replace("__DATA__", json.dumps(payload)))
    print("wrote", path)
    print(payload["desc"])


if __name__ == "__main__":
    main()

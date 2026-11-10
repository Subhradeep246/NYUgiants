"""
Builds index.html: the race viewer itself, with a play picker.
Picks the most dramatic pressured plays (receiver won the race, completed, big gain, plus a few where the
open man was ignored and a few sacks) and embeds their tracking so the page is a single static file.

Usage: python build_site.py [--n 24]
"""
import argparse
import json
import os

import pandas as pd

from race_tv import build, OUT, DATA

HERE = os.path.dirname(os.path.abspath(__file__))


def pick_plays(n):
    r = pd.read_csv(os.path.join(OUT, "race_routes.csv"))
    r["is_target"] = r.targeted.fillna(False).astype(bool)      # nflverse, by player ID (compute_race.py)
    r["yds"] = r.playDescription.str.extract(r"for (-?\d+) yards").astype(float)
    r["td"] = r.playDescription.str.contains("TOUCHDOWN")
    have = {int(f.split("_")[1].split(".")[0]) for f in os.listdir(os.path.join(DATA, "tracking"))}
    p = r[r.pressured & r.gameId.isin(have) & ~r.playDescription.str.contains("No Play|PENALTY", regex=True)]

    # A) target won the race, completed, long gain
    a = p[p.is_target & p.won & p.passResult.eq("C") & p.officialPosition.isin(["WR", "TE"])]
    a = a.sort_values(["td", "yds"], ascending=False).drop_duplicates(["gameId", "playId"]).head(n // 2)
    # B) open man ignored: someone won by >= 0.5s, target did not win, incomplete
    plays_won = p[p.won & (p.margin >= 0.5)][["gameId", "playId"]].drop_duplicates()
    b = p[p.is_target & ~p.won & p.passResult.eq("I")].merge(plays_won, on=["gameId", "playId"])
    b = b.drop_duplicates(["gameId", "playId"]).sample(min(len(b), n // 4), random_state=3)
    # C) sacks where nobody won
    plays_nowin = p.groupby(["gameId", "playId"]).won.sum().rename("w").reset_index()
    c = p[p.passResult.eq("S")].merge(plays_nowin[plays_nowin.w == 0][["gameId", "playId"]], on=["gameId", "playId"])
    c = c.drop_duplicates(["gameId", "playId"]).sample(min(len(c), n - len(a) - len(b)), random_state=3)

    out = []
    for kind, d in [("Won the race", a), ("Open man not thrown to", b), ("Nobody won: sack", c)]:
        for x in d.itertuples():
            out.append(dict(game=int(x.gameId), play=int(x.playId), kind=kind))
    return out


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>The Race: pocket clock vs route clock</title>
<style>
:root{--ink:#111;--muted:#666;--line:#e5e5e5;--red:#d33;--green:#1a8a3a;--accent:#111}
*{box-sizing:border-box}body{margin:0;background:#fff;color:var(--ink);font:16px/1.6 Georgia,"Times New Roman",serif}
main{max-width:860px;margin:0 auto;padding:48px 20px 80px}
h1{font-size:30px;font-weight:normal;margin:0 0 6px}h2{font-size:19px;font-weight:normal;margin:40px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
p{margin:0 0 14px}.sub{color:var(--muted);margin:0 0 28px}a{color:var(--ink)}
.ui{font:13px/1.4 -apple-system,Helvetica,Arial,sans-serif}
.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
select{font:inherit;padding:5px 6px;border:1px solid #ccc;background:#fff;max-width:100%;flex:1}
button{font:inherit;padding:5px 12px;border:1px solid #ccc;background:#fff;cursor:pointer}button:hover{border-color:#111}
label{color:var(--muted)}
canvas{width:100%;height:auto;display:block;border:1px solid var(--line)}
.l3{display:grid;grid-template-columns:150px 1fr 1fr;gap:16px;align-items:start;padding:12px 0;border-bottom:1px solid var(--line)}
.big{font-size:28px;font-variant-numeric:tabular-nums;line-height:1.1}.lab{color:var(--muted);font-size:12px}
.rows{display:flex;flex-direction:column;gap:4px}.row{display:grid;grid-template-columns:110px 1fr 56px;gap:8px;align-items:center}
.trk{height:6px;background:#f0f0f0;position:relative}.trk i{position:absolute;left:0;top:0;bottom:0;width:0}
.row.pocket .trk i{background:var(--red)}.row.route .trk i{background:var(--green)}.tick{position:absolute;top:-3px;bottom:-3px;width:1px;background:#111}
.row .t{text-align:right;font-variant-numeric:tabular-nums;color:var(--muted)}.row.won .t{color:var(--green)}.row.pocket.won .t{color:var(--red)}
.verdict{font-size:15px;min-height:20px}.call{color:var(--muted);margin-top:4px}
.ctl{display:flex;gap:10px;align-items:center;padding:8px 0;color:var(--muted)}.ctl input{flex:1}
table{border-collapse:collapse;width:100%;margin:6px 0 16px;font-size:15px}td{padding:6px 10px 6px 0;border-bottom:1px solid var(--line);vertical-align:top}td:first-child{color:var(--muted);width:42%}
figure{margin:18px 0}figure img{width:100%;display:block;border:1px solid var(--line)}figcaption{color:var(--muted);font-size:14px;margin-top:4px}
code,.k{font:13px ui-monospace,Menlo,monospace}footer{margin-top:48px;color:var(--muted);font-size:14px}
@media(max-width:640px){.l3{grid-template-columns:1fr}}
</style></head><body><main>
<h1>The Race</h1>
<p class="sub">Pocket clock vs. route clock on every 2021 pass play (weeks 1&ndash;8). NFL Big Data Bowl, NYU Edition.</p>
<p>Two clocks start at the snap. The <span style="color:var(--red)">pocket clock</span> stops when a pass rusher gets within 2 yards of the quarterback. Each <span style="color:var(--green)">route clock</span> stops when that receiver has 3 yards of separation after a 5-yard stem. A receiver wins if he is open before the pocket breaks. The black tick is the throw.</p>

<div class="ui">
<div class="bar"><select id="pick"></select><button id="play">Play</button><button id="reset">Restart</button><label><input type="checkbox" id="an"> separation lines</label></div>
<canvas id="field" width="1400" height="620"></canvas>
<div class="ctl"><input type="range" id="scrub" min="0" value="0" step="1"><span id="tt">0.0 s</span></div>
<div class="l3"><div><div class="big" id="clk">0.0<span style="font-size:14px;color:var(--muted)"> s</span></div><div class="lab" id="sit"></div></div>
<div class="rows" id="rows"></div><div><div class="verdict" id="ver">&nbsp;</div><div class="lab" id="desc" style="margin-top:6px"></div><div class="call" id="call"></div></div></div>
</div>

<h2>What we found</h2>
<table>
<tr><td>Pressured dropbacks</td><td>2,660. The pocket breaks at a median 2.7 s; the first receiver is open at 1.6 s.</td></tr>
<tr><td>Throw to a receiver who won vs. lost his race</td><td>57% vs. 48% completion, +6.4 points above expected (CPOE), same throw depth (11.0 vs. 10.8 air yards). n = 1,593.</td></tr>
<tr><td>Short vs. deep</td><td>The gap is on short passes (62% vs. 49%). On deep passes there is none (43% vs. 45%).</td></tr>
<tr><td>Sacks</td><td>30% when nobody wins, 17% when a receiver wins by more than a second.</td></tr>
<tr><td>Who wins</td><td>Marquise Brown 86%, Devin Duvernay 83%. Davante Adams 40% and Ja'Marr Chase 41%, though both still produce high EPA per target.</td></tr>
<tr><td>Extra blocker vs. man coverage</td><td>Buys no pocket time (2.6 &rarr; 2.5 s) and cuts the race-won rate from 88% to 83%.</td></tr>
</table>
<figure><img src="output/chart_validation_target.png" alt="Completion when the target won vs lost the race"><figcaption>Completion and completion above expected, by race outcome.</figcaption></figure>
<figure><img src="output/chart_receivers.png" alt="Receivers: route clock vs win rate"><figcaption>One dot per receiver: how fast he gets open vs. how often he beats the pocket.</figcaption></figure>
<figure><img src="output/chart_value.png" alt="Win rate vs EPA and NGS separation"><figcaption>Winning early is not the same as being valuable, and it is not NGS separation.</figcaption></figure>
<figure><img src="output/chart_protection.png" alt="Protection exchange rate"><figcaption>What keeping a blocker in actually buys, by coverage.</figcaption></figure>

<h2>Limits</h2>
<p>The race measures timing, not value: it adds almost nothing to completion models beyond separation at the throw. Matchups are the nearest defender, the 2/3/5-yard thresholds are not tuned, and tracking stops at the throw. EPA, CPOE and win probability come from nflverse.</p>

<footer>Code and method: <a href="README.md">README</a> &middot; <a href="https://github.com/Subhradeep246/NYUgiants">GitHub</a> &middot; <a href="output/the_race.mp4">1-minute video</a></footer>
</main>
<script>
const PLAYS = __PLAYS__;
const TH = PLAYS[0].thresholds;
const cv = document.getElementById('field'), ctx = cv.getContext('2d');
const X = x => x / 120 * cv.width, Y = y => (53.3 - y) / 53.3 * cv.height;
const color = p => p.role === 'Pass' ? '#e6b800' : p.role === 'Pass Route' ? '#1a8a3a' : p.role === 'Pass Rush' ? '#dd3333' : p.role === 'Coverage' ? '#ffffff' : '#9a9a9a';
let D, F, n, snapK, endK, tPressure, tEnd, maxT, k = 0, timer = null;
const t = k => Math.max(0, (k - snapK) / 10);
const pick = document.getElementById('pick');
PLAYS.forEach((p, i) => { const o = document.createElement('option'); o.value = i; o.textContent = `${p.kind} — ${p.title.split('·')[0].trim()} — ${p.desc.replace(/^\(\d+:\d+\)\s*/, '').replace(/\(Shotgun\)\s*/, '').slice(0, 80)}`; pick.appendChild(o); });
pick.onchange = () => load(+pick.value);

function load(i) {
  stop(); D = PLAYS[i]; F = D.frames; n = F.length; snapK = F.indexOf(D.f_snap); endK = F.indexOf(D.f_end);
  const minK = snapK + TH.min_frames; tPressure = null;
  for (let j = minK; j <= endK; j++) if (D.pocket[j] != null && D.pocket[j] <= TH.pressure) { tPressure = t(j); break; }
  D.receivers.forEach(r => { r.tOpen = null; for (let j = minK; j <= endK; j++) if (r.sep[j] >= TH.open && r.moved[j] >= TH.route) { r.tOpen = t(j); break; } });
  D.receivers.sort((a, b) => (a.tOpen ?? 99) - (b.tOpen ?? 99));
  tEnd = t(endK); maxT = Math.max(3.5, tEnd + 0.3);
  document.getElementById('sit').textContent = D.title.replace(/\s+·\s+/g, ' · ') + ' · ' + D.situation + (D.value ? ' · ' + D.value.replace(/\s+·\s+/g, ' · ') : '');
  document.getElementById('desc').textContent = D.desc;
  const R = document.getElementById('rows'); R.innerHTML = '';
  const row = (cls, label) => { const el = document.createElement('div'); el.className = 'row ' + cls; el.innerHTML = `<div>${label}</div><div class="trk"><i></i><span class="tick" style="left:${100 * tEnd / maxT}%"></span></div><div class="t">—</div>`; R.appendChild(el); return el; };
  D.pocketEl = row('pocket', 'POCKET ' + D.defense);
  D.receivers.slice(0, 4).forEach(r => r.el = row('route', r.name.split(' ').slice(-1)[0].toUpperCase()));
  document.getElementById('call').textContent = D.call || '';
  k = 0; const s = document.getElementById('scrub'); s.max = n - 1; s.value = 0; draw(0);
}
function drawField() {
  ctx.fillStyle = '#5b8a64'; ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.fillStyle = '#4e7a57'; ctx.fillRect(0, 0, X(10), cv.height); ctx.fillRect(X(110), 0, X(10), cv.height);
  ctx.strokeStyle = 'rgba(255,255,255,.3)'; ctx.lineWidth = 1;
  for (let x = 10; x <= 110; x += 5) { ctx.beginPath(); ctx.moveTo(X(x), 0); ctx.lineTo(X(x), cv.height); ctx.stroke(); }
  ctx.fillStyle = 'rgba(255,255,255,.55)'; ctx.font = 'bold 14px sans-serif'; ctx.textAlign = 'center';
  for (let x = 20; x <= 100; x += 10) { const yd = x <= 60 ? x - 10 : 110 - x; ctx.fillText(yd, X(x), 22); }
  ctx.strokeStyle = '#4da3ff'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(X(D.los), 0); ctx.lineTo(X(D.los), cv.height); ctx.stroke();
}
function draw(k) {
  drawField();
  const q = D.players.find(p => p.role === 'Pass'); const lines = document.getElementById('an').checked;
  if (q && q.xy[k][0] != null) { ctx.beginPath(); ctx.arc(X(q.xy[k][0]), Y(q.xy[k][1]), X(TH.pressure), 0, 7); ctx.strokeStyle = 'rgba(229,72,77,.6)'; ctx.setLineDash([4, 4]); ctx.stroke(); ctx.setLineDash([]); }
  if (lines) D.receivers.forEach(r => { const p = D.players.find(p => p.id === r.id); if (!p || p.xy[k][0] == null) return;
    let best = null, bd = 1e9; D.players.filter(d => d.role === 'Coverage').forEach(d => { if (d.xy[k][0] == null) return; const dd = Math.hypot(d.xy[k][0] - p.xy[k][0], d.xy[k][1] - p.xy[k][1]); if (dd < bd) { bd = dd; best = d; } });
    if (best) { ctx.beginPath(); ctx.moveTo(X(p.xy[k][0]), Y(p.xy[k][1])); ctx.lineTo(X(best.xy[k][0]), Y(best.xy[k][1])); ctx.strokeStyle = bd >= TH.open ? 'rgba(63,185,80,.85)' : 'rgba(255,255,255,.35)'; ctx.lineWidth = 2; ctx.stroke(); } });
  D.players.forEach(p => { const [x, y] = p.xy[k]; if (x == null) return;
    ctx.beginPath(); ctx.arc(X(x), Y(y), 10, 0, 7); ctx.fillStyle = color(p); ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = '#0e1116'; ctx.stroke();
    ctx.fillStyle = '#0e1116'; ctx.font = 'bold 10px sans-serif'; ctx.textAlign = 'center'; ctx.fillText(p.jersey, X(x), Y(y) + 4);
    if (p.role === 'Pass Route') { ctx.fillStyle = '#fff'; ctx.font = '600 12px sans-serif'; ctx.fillText(p.name.split(' ').slice(-1)[0], X(x), Y(y) - 15); } });
  const b = D.ball[k]; if (b && b[0] != null) { ctx.beginPath(); ctx.ellipse(X(b[0]), Y(b[1]), 6, 4, 0, 0, 7); ctx.fillStyle = '#8b4513'; ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.stroke(); }
  const now = t(k), pre = k < snapK;
  const set = (el, prog, label, won) => { el.querySelector('i').style.width = (100 * Math.min(prog, maxT) / maxT) + '%'; el.querySelector('.t').textContent = label; el.classList.toggle('won', won); };
  const pw = tPressure != null && now >= tPressure;
  set(D.pocketEl, pre ? 0 : (pw ? tPressure : now), pre ? '—' : pw ? `${tPressure.toFixed(1)} s` : `${D.pocket[k] != null ? D.pocket[k].toFixed(1) + ' yd' : ''}`, pw);
  D.receivers.slice(0, 4).forEach(r => { const w = r.tOpen != null && now >= r.tOpen; set(r.el, pre ? 0 : (w ? r.tOpen : now), pre ? '—' : w ? `${r.tOpen.toFixed(1)} s` : `${r.sep[k].toFixed(1)} yd`, w); });
  const tx = (pre ? '-' : '') + Math.abs((k - snapK) / 10).toFixed(1);
  document.getElementById('clk').firstChild.textContent = tx; document.getElementById('tt').textContent = tx + ' s';
  const v = document.getElementById('ver');
  if (pre) v.textContent = 'pre-snap';
  else if (k >= endK) { const w = D.receivers.filter(r => r.tOpen != null && (tPressure == null || r.tOpen < tPressure));
    v.textContent = tPressure == null ? `Clean pocket for ${tEnd.toFixed(1)} s, ${w.length} open` : w.length ? `${w.map(r => r.name.split(' ').slice(-1)[0]).join(', ')} won by ${(tPressure - w[0].tOpen).toFixed(1)} s` : `Pocket broke at ${tPressure.toFixed(1)} s, nobody open in time`; }
  else v.textContent = pw ? 'Pocket broken' : 'Racing';
}
function step() { k++; if (k >= n) { k = n - 1; stop(); } document.getElementById('scrub').value = k; draw(k); }
function play() { if (timer) return stop(); if (k >= n - 1) k = 0; timer = setInterval(step, 100); document.getElementById('play').textContent = 'Pause'; }
function stop() { clearInterval(timer); timer = null; document.getElementById('play').textContent = 'Play'; }
document.getElementById('play').onclick = play; document.getElementById('reset').onclick = () => { stop(); k = 0; document.getElementById('scrub').value = 0; draw(0); };
document.getElementById('scrub').oninput = e => { stop(); k = +e.target.value; draw(k); };
document.getElementById('an').onchange = () => draw(k);
load(0);
</script></body></html>
"""


def main(n):
    picks = pick_plays(n)
    calls = {}
    cpath = os.path.join(OUT, "commentary.csv")
    if os.path.exists(cpath):
        for r in pd.read_csv(cpath).itertuples():
            calls[(int(r.gameId), int(r.playId))] = r.call
    payloads = []
    for p in picks:
        d = build(p["game"], p["play"])
        d["kind"] = p["kind"]
        d["call"] = calls.get((p["game"], p["play"]), "")
        # trim payload: round coordinates
        for pl in d["players"]:
            pl["xy"] = [[None, None] if (a is None or a != a) else [round(a, 2), round(b, 2)] for a, b in pl["xy"]]
        d["ball"] = [[None, None] if (a is None or a != a) else [round(a, 2), round(b, 2)] for a, b in d["ball"]]
        d["pocket"] = [None if v is None else round(v, 2) for v in d["pocket"]]
        for r in d["receivers"]:
            r["sep"] = [round(v, 2) for v in r["sep"]]; r["moved"] = [round(v, 2) for v in r["moved"]]
        payloads.append(d)
        print(p["kind"], "|", d["desc"][:90])
    html = HTML.replace("__PLAYS__", json.dumps(payloads))
    with open(os.path.join(HERE, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote index.html with {len(payloads)} plays, {len(html)//1024} KB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=24)
    main(ap.parse_args().n)

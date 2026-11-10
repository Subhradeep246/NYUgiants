"""
Builds index.html: the race viewer itself, with a play picker.
Picks the most dramatic pressured plays (receiver won the race, completed, big gain, plus a few where the
open man was ignored and a few sacks) and embeds their tracking so the page is a single static file.

Usage: python build_site.py [--n 24]
"""
import argparse
import json
import os
import re

import pandas as pd

from race_tv import build, OUT, DATA

HERE = os.path.dirname(os.path.abspath(__file__))


def target_of(desc):
    m = re.search(r"pass .*?to ([A-Z][A-Za-z'\-]*\.\s?[A-Za-z'\-\.]+)", str(desc))
    return m.group(1).replace(" ", "").lower() if m else None


def pick_plays(n):
    r = pd.read_csv(os.path.join(OUT, "race_routes.csv"))
    r["short"] = r.displayName.map(lambda s: f"{s.split()[0][0]}.{s.split()[-1]}".lower())
    r["tgt"] = r.playDescription.map(target_of)
    r["yds"] = r.playDescription.str.extract(r"for (-?\d+) yards").astype(float)
    r["td"] = r.playDescription.str.contains("TOUCHDOWN")
    have = {int(f.split("_")[1].split(".")[0]) for f in os.listdir(os.path.join(DATA, "tracking"))}
    p = r[r.pressured & r.gameId.isin(have) & ~r.playDescription.str.contains("No Play|PENALTY", regex=True)]

    # A) target won the race, completed, long gain
    a = p[(p.tgt == p.short) & p.won & p.passResult.eq("C") & p.officialPosition.isin(["WR", "TE"])]
    a = a.sort_values(["td", "yds"], ascending=False).drop_duplicates(["gameId", "playId"]).head(n // 2)
    # B) open man ignored: someone won by >= 0.5s, target did not win, incomplete
    plays_won = p[p.won & (p.margin >= 0.5)][["gameId", "playId"]].drop_duplicates()
    b = p[(p.tgt == p.short) & ~p.won & p.passResult.eq("I")].merge(plays_won, on=["gameId", "playId"])
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
:root{--bg:#0e1116;--panel:#151a21;--ink:#e8eaed;--muted:#8b949e;--line:#262d37;--red:#e5484d;--green:#3fb950;--accent:#f0c35a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.bar{display:flex;align-items:center;gap:14px;padding:10px 16px;border-bottom:1px solid var(--line);background:var(--panel);flex-wrap:wrap}
.bar h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.01em}.bar h1 span{color:var(--muted);font-weight:400;margin-left:8px}
select{background:#0e1116;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:7px 10px;font-size:14px;max-width:560px}
.bar .r{margin-left:auto;display:flex;gap:8px;align-items:center;color:var(--muted);font-size:13px}
button{background:#0e1116;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:7px 12px;font-size:13px;cursor:pointer}button:hover{border-color:var(--accent)}
.stage{position:relative;max-width:1400px;margin:0 auto}
canvas{width:100%;height:auto;display:block;background:#1d4d2b;max-height:62vh;object-fit:cover}
.l3{margin:10px 16px 0;background:rgba(10,13,18,.92);border:1px solid var(--line);border-left:5px solid var(--accent);border-radius:6px;padding:10px 16px;display:grid;grid-template-columns:190px 1fr 1.1fr;gap:16px;align-items:center}
.l3 .big{font-size:36px;font-weight:700;font-variant-numeric:tabular-nums;line-height:1}.l3 .lab{font-size:11px;color:var(--muted);letter-spacing:.06em;text-transform:uppercase}
.rows{display:flex;flex-direction:column;gap:5px}.row{display:grid;grid-template-columns:130px 1fr 80px;gap:10px;align-items:center;font-size:13px;font-weight:600}
.trk{height:9px;background:#232a34;border-radius:3px;position:relative;overflow:hidden}.trk i{position:absolute;left:0;top:0;bottom:0;width:0;border-radius:3px}
.row.pocket .trk i{background:var(--red)}.row.route .trk i{background:var(--green)}.tick{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--accent)}
.row .t{text-align:right;font-variant-numeric:tabular-nums;color:var(--muted);font-weight:600}.row.won .t{color:var(--green)}.row.pocket.won .t{color:var(--red)}
.verdict{font-size:18px;font-weight:700;color:var(--accent);min-height:24px}.call{font-size:13px;color:var(--ink);margin-top:6px;opacity:.9}
.ctl{display:flex;gap:10px;align-items:center;padding:10px 16px;max-width:1400px;margin:0 auto;color:var(--muted);font-size:13px}.ctl input{flex:1}
.notes{max-width:1000px;margin:0 auto;padding:26px 16px 50px;color:var(--muted)}.notes h2{color:var(--ink);font-size:16px;margin:22px 0 6px}.notes p{margin:0 0 10px;max-width:760px}
.notes table{border-collapse:collapse;font-size:13px;margin:8px 0 14px}.notes td,.notes th{padding:5px 12px 5px 0;text-align:left;border-bottom:1px solid var(--line)}.notes th{color:var(--muted);font-weight:500}
.notes a{color:var(--ink)}.imgs{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px;margin-top:10px}.imgs img{width:100%;border:1px solid var(--line);border-radius:4px;background:#fff}
.k{display:inline-block;background:#232a34;border-radius:4px;padding:1px 6px;font-family:ui-monospace,Menlo,monospace;font-size:12px}
</style></head><body>
<div class="bar"><h1>The Race<span>pocket clock vs route clock · 2021 Next Gen Stats tracking</span></h1>
<select id="pick"></select>
<div class="r"><button id="play">Play</button><button id="reset">Restart</button><label><input type="checkbox" id="an"> show separation lines</label></div></div>
<div class="stage"><canvas id="field" width="1400" height="560"></canvas>
<div class="l3"><div><div class="lab" id="sit"></div><div class="big" id="clk">0.0<span style="font-size:16px;color:var(--muted)"> s</span></div><div class="lab" id="desc" style="text-transform:none;letter-spacing:0;margin-top:6px;line-height:1.3"></div></div>
<div class="rows" id="rows"></div><div><div class="lab">verdict</div><div class="verdict" id="ver">&nbsp;</div><div class="call" id="call"></div></div></div></div>
<div class="ctl"><input type="range" id="scrub" min="0" value="0" step="1"><span id="tt">0.0 s</span></div>

<div class="notes">
<h2>What you're looking at</h2>
<p>Two clocks start at the snap. The red one measures how long until a pass rusher gets within 2 yards of the quarterback. Each green one measures how long until that receiver has 3 yards of separation from the nearest defender after travelling 5 yards from where he lined up, so the cushion he was given before the snap doesn't count. The gold tick is the throw. A receiver wins when his clock beats the red one.</p>
<p>Plays in the menu are grouped: receivers who won the race and were thrown to; plays where someone won by half a second or more but the ball went elsewhere; and sacks where nobody won.</p>
<h2>Why it matters</h2>
<p>On 1,290 pressured dropbacks where the target can be read from the play description, throws to a receiver who had already won his race were completed 73% of the time; throws to one who hadn't, 60%. Holding route depth fixed, the gap is 77% vs 61% on short passes and nearly nothing on deep ones. Sack rate on pressured dropbacks is 30% when no receiver wins and 17% when one wins by more than a second.</p>
<p>We also checked whether the race adds anything a separation stat does not. In a logistic regression on the same passes, separation at the throw is the strongest predictor of completion and the race margin adds nothing on top of it. We think that is the right result: the race is the same information measured earlier and attributed to a player and a moment, not a new predictor. The thresholds (2, 3 and 5 yards) move the levels but not the story; the sack-rate gap holds at every setting we tried, the completion gap shrinks when the pressure radius is tightened. Details in <span class="k">output/model_results.csv</span> and <span class="k">output/robustness.csv</span>.</p>
<table><tr><th>Pressured dropbacks</th><th>Pocket breaks (median)</th><th>First receiver open (median)</th><th>WR / TE / RB already open when it breaks</th></tr>
<tr><td>2,649</td><td>2.7 s</td><td>1.6 s</td><td>63% / 57% / 71%</td></tr></table>
<h2>Who wins the race</h2>
<p>Baltimore's receivers, by a distance: Marquise Brown wins 86% of his pressured routes, Devin Duvernay 83%, James Proche 82%, Mark Andrews 77%. The quick game shows up in the numbers. At the other end sit Davante Adams (40%) and Ja'Marr Chase (41%), because deep routes come open after the pocket has already broken. This is a measure of who gives the quarterback an early answer, not of who the best receiver is.</p>
<h2>For coaches</h2>
<p>Keeping a back or tight end in to block is supposed to buy time. Against man coverage it doesn't: one extra blocker leaves the pocket clock at 2.5 s (2.6 s with none) and drops the share of plays with someone open in time from 88% to 83%. Against zone, two extra blockers buy 0.6 s. The per-team numbers are in <span class="k">output/protection_exchange.csv</span>; every defense's pocket clock and completion allowed by coverage call is in <span class="k">output/defense_cards.csv</span>.</p>
<div class="imgs"><img src="output/chart_validation_target.png" alt=""><img src="output/chart_receivers.png" alt=""><img src="output/chart_protection.png" alt=""><img src="output/chart_teams.png" alt=""></div>
<h2>Method, limits, code</h2>
<p>Python with pandas, numpy and matplotlib; everything runs from the public Big Data Bowl files. The target is parsed from the play text, so interceptions (worded differently) are left out of the validation. Matchups are the nearest defender; PFF does not label assignments. Thresholds are football-reasonable, not tuned. Tracking stops at the throw, so this describes the race up to the release and says nothing about the catch. See the <a href="README.md">README</a> and <span class="k">compute_race.py</span>, <span class="k">team_report.py</span>, <span class="k">race_tv.py</span>.</p>
</div>

<script>
const PLAYS = __PLAYS__;
const TH = PLAYS[0].thresholds;
const cv = document.getElementById('field'), ctx = cv.getContext('2d');
const X = x => x / 120 * cv.width, Y = y => (53.3 - y) / 53.3 * cv.height;
const color = p => p.role === 'Pass' ? '#f0c35a' : p.role === 'Pass Route' ? '#3fb950' : p.role === 'Pass Rush' ? '#e5484d' : p.role === 'Coverage' ? '#ffffff' : '#8b949e';
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
  document.getElementById('sit').textContent = D.title.replace(/\s+·\s+/g, ' · ') + ' · ' + D.situation;
  document.getElementById('desc').textContent = D.desc;
  const R = document.getElementById('rows'); R.innerHTML = '';
  const row = (cls, label) => { const el = document.createElement('div'); el.className = 'row ' + cls; el.innerHTML = `<div>${label}</div><div class="trk"><i></i><span class="tick" style="left:${100 * tEnd / maxT}%"></span></div><div class="t">—</div>`; R.appendChild(el); return el; };
  D.pocketEl = row('pocket', 'POCKET ' + D.defense);
  D.receivers.slice(0, 4).forEach(r => r.el = row('route', r.name.split(' ').slice(-1)[0].toUpperCase()));
  document.getElementById('call').textContent = D.call || '';
  k = 0; const s = document.getElementById('scrub'); s.max = n - 1; s.value = 0; draw(0);
}
function drawField() {
  ctx.fillStyle = '#1d4d2b'; ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.fillStyle = '#173d22'; ctx.fillRect(0, 0, X(10), cv.height); ctx.fillRect(X(110), 0, X(10), cv.height);
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

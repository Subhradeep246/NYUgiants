"""
THE RACE - AI commentary
------------------------
Turns one play's race numbers into a broadcaster's one-liner using a local LLM (Qwen via Ollama).
Falls back to a rule-based line if Ollama is not running, so the script always works.

Usage:
  python race_commentary.py --game 2021090900 --play 687            # one play, prints the call
  python race_commentary.py --game 2021090900 --play 687 --inject   # also writes the call into the TV mock HTML
  python race_commentary.py --sample 8                               # 8 random pressured plays -> output/commentary.csv
Options: --model qwen3:4b   (any model you have pulled in Ollama)
"""
import argparse
import json
import os
import re          # used to strip qwen3 <think> tags and old AI-call panels
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
OLLAMA = "http://localhost:11434/api/generate"

SYSTEM = ("You are an NFL TV analyst. You get the timing numbers of one pass play: when the pocket broke "
          "(pocket clock) and when each receiver got open (route clock), plus what the play was worth "
          "(epa, win_prob_added_pct). Write ONE punchy sentence (max 25 words) a broadcaster would say on air. "
          "Be concrete: name players and quote the seconds; mention the win-probability swing only if it is 3 points "
          "or more. Never invent facts.")


def play_facts(game, play):
    r = pd.read_csv(os.path.join(OUT, "race_routes.csv"))
    r = r[(r.gameId == game) & (r.playId == play)].sort_values("t_open")
    if r.empty:
        raise SystemExit("play not found in output/race_routes.csv")
    p = r.iloc[0]
    tp = None if pd.isna(p.t_pressure) else float(p.t_pressure)
    desc = str(p.playDescription)
    recs = [dict(name=x.displayName, pos=x.officialPosition,
                 t_open=None if pd.isna(x.t_open) else round(float(x.t_open), 1),
                 won=bool(x.won), sep_at_throw=round(float(x.sep_at_throw), 1),
                 targeted=bool(x.targeted) if not pd.isna(x.targeted) else False)   # nflverse, by player ID
            for x in r.itertuples()]
    num = lambda v, d: None if pd.isna(v) else round(float(v), d)
    return dict(offense=p.possessionTeam, defense=p.defensiveTeam, down=int(p.down), yards_to_go=int(p.yardsToGo),
                coverage=p.pff_passCoverage, result={"C": "complete", "I": "incomplete", "S": "sack", "IN": "interception", "R": "scramble"}.get(p.passResult, p.passResult),
                pocket_clock_s=tp, ball_out_s=round(float(p.play_len), 1), receivers=recs,
                epa=num(p.epa, 2), win_prob_added_pct=None if pd.isna(p.wpa) else round(100 * float(p.wpa), 1),
                play_text=desc)


def template_call(f):
    winners = [r for r in f["receivers"] if r["won"]]
    tgt = next((r for r in f["receivers"] if r["targeted"]), None)
    pc = f["pocket_clock_s"]
    if pc is None:
        return f"Clean pocket for {f['ball_out_s']}s — {len(winners)} receivers open, and the {f['offense']} take their pick."
    if not winners:
        return f"Pocket gone at {pc}s and nobody had won their route — that's a {f['result']} waiting to happen."
    w = winners[0]
    line = f"{w['name'].split()[-1]} wins the race — open at {w['t_open']}s, pocket breaks at {pc}s"
    if tgt and not tgt["won"]:
        line += f", but the ball goes to {tgt['name'].split()[-1]}, who hadn't shaken his man."
    elif tgt and tgt["won"]:
        line += f" — and the quarterback finds him. {f['result'].capitalize()}."
    else:
        line += "."
    if f.get("win_prob_added_pct") is not None and abs(f["win_prob_added_pct"]) >= 3:
        line += f" That swings win probability {f['win_prob_added_pct']:+.1f}%."
    return line


def llm_call(f, model):
    prompt = SYSTEM + "\n\nPLAY FACTS (JSON):\n" + json.dumps(f, indent=1) + "\n\nOne-sentence on-air call:"
    req = urllib.request.Request(OLLAMA, data=json.dumps(dict(model=model, prompt=prompt, stream=False,
                                                              options=dict(temperature=0.7, num_predict=80))).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        out = json.loads(resp.read())["response"].strip()
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()      # qwen3 thinking tags
    return out.split("\n")[0].strip().strip('"')


def commentary(game, play, model):
    f = play_facts(game, play)
    try:
        line, source = llm_call(f, model), model
    except Exception as e:                                               # Ollama not running / model missing
        line, source = template_call(f), "rule-based call — run with Ollama for the LLM version"
    return f, line, source


def inject(game, play, line, source):
    path = os.path.join(OUT, f"race_tv_{game}_{play}.html")
    if not os.path.exists(path):
        print("TV mock not found; run race_tv.py first:", path); return
    html = open(path, encoding="utf-8").read()
    panel = (f'<div class="banner" id="aicall" style="color:#f3f5f7;font-weight:600;font-size:16px">🎙 <b style="color:#ffd166">AI CALL</b> '
             f'<span style="color:#9aa5b1;font-size:12px">({source})</span><br>{line}</div>')
    html = re.sub(r'<div class="banner" id="aicall".*?</div>', "", html, flags=re.S)
    html = html.replace('<div class="banner" id="banner"></div>', '<div class="banner" id="banner"></div>' + panel)
    open(path, "w", encoding="utf-8").write(html)
    print("injected into", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int); ap.add_argument("--play", type=int)
    ap.add_argument("--model", default="qwen3:4b"); ap.add_argument("--inject", action="store_true")
    ap.add_argument("--sample", type=int, default=0)
    a = ap.parse_args()
    if a.sample:
        r = pd.read_csv(os.path.join(OUT, "race_routes.csv"))
        plays = r[r.pressured].drop_duplicates(["gameId", "playId"]).sample(a.sample, random_state=7)
        rows = []
        for p in plays.itertuples():
            f, line, src = commentary(int(p.gameId), int(p.playId), a.model)
            rows.append(dict(gameId=p.gameId, playId=p.playId, play=f["play_text"], call=line, source=src))
            print(f"\n{f['play_text']}\n  🎙 {line}   [{src}]")
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "commentary.csv"), index=False)
    else:
        f, line, src = commentary(a.game, a.play, a.model)
        print(f["play_text"]); print("🎙", line, f"[{src}]")
        if a.inject:
            inject(a.game, a.play, line, src)

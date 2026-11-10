# THE RACE — Pocket Clock vs. Route Clock.

> **Site:** `index.html` is the viewer itself — 24 plays (race won, open man not thrown to, sacks) with the two clocks, a play picker and the findings underneath. Single-play mocks are in `output/race_tv_*.html`.

![Winning the race changes completion probability](output/chart_validation_target.png)
![Model](output/chart_model.png)

**What we built.** Every pass play is a race between two clocks: how fast the pass rush breaks the pocket, and how fast a receiver gets open. Using the 2021 Next Gen Stats tracking data (weeks 1–8, 122 games) joined to PFF's role labels, we compute both clocks on every dropback — the **pocket clock** (seconds from snap until a rusher is within 2 yds of the QB) and a **route clock** for every WR/TE/RB (seconds until he has 3 yds of separation from the nearest coverage defender *after* travelling 5 yds, so pre-snap cushion doesn't count) — and score who won.

**What it reveals.** Across 2,649 pressured dropbacks the pocket breaks at a median 2.7 s while the median route needs ~1.8 s to come open — yet when pressure arrives only 63% of WR routes (57% of TE, 71% of RB) have already won. The *Win-Rate-Under-Pressure* leaderboard separates receivers sharply: Baltimore's quick-game receivers (Marquise Brown 86%, Duvernay 83%, Proche 82%, Mark Andrews 77%) dominate and BAL is also #1 in *Protection–Route Sync* (first receiver open 1.4 s before the pocket breaks), while deep-route stars Davante Adams (40%) and Ja'Marr Chase (41%) sit at the bottom — the metric finds *hot-read* receivers who win early, not the best receivers overall. Existing stats measure *how much* separation a receiver gets at the throw; this measures *when* he gets it relative to the protection — the first public metric to put pass rush and route running on one clock, which became possible only when this dataset paired PFF rush/route roles with receiver tracking.

**Does winning the race matter? (validation, `team_report.py` and `model.py`)** On 1,290 pressured, targeted passes, throwing to a receiver who had won his race completes **73%** of the time vs **60%** when the target was not yet open. Controlling for route depth the gap holds where the race is actually decided: on short passes **77% vs 61%** (n=994); on deep passes it nearly vanishes (60 vs 58). Sack rate on pressured dropbacks falls from 30% when nobody wins to 17% when a receiver wins by more than a second. We then asked the harder question with a logistic regression (numpy, 5-fold CV): does the race margin predict completion *beyond* what separation at the throw already tells you? It does not (AUC 0.679 vs 0.677) — separation at the throw is the strongest single predictor (0.688 alone), and the margin adds only a little to pre-throw context (0.594 → 0.601). Our reading: the race is the **timing decomposition of separation** — the same information, but available before the throw and attributable to a player and a moment. That is what makes it usable on a broadcast and in a game plan; it is not a new predictor of completion.

**Robustness (`model.py`, 30-game subset).** Thresholds move the levels but not the story: the median pocket clock stays 2.5–2.8 s and first-open 1.7 s across pressure radii of 1.5–2.5 yd and open thresholds of 2.5–3.5 yd; the sack-rate gap (nobody won vs someone won) persists in every setting; the completion gap for the targeted receiver is 7 pts at the baseline, 4 pts at a 2.5 yd radius and disappears at 1.5 yd (where only 19% of plays count as pressured and n=154). Player rankings on 30 games are noisy and should be read from the full 8 weeks. Table: `output/robustness.csv`.

**For teams (`team_report.py`).** *Protection exchange rate*: keeping extra RB/TEs in to block is measured as pocket time bought vs. route clock lost. League-wide, against **man** coverage one extra blocker buys **no** pocket time (2.6 → 2.5 s) while cutting race-won rate from 88% to 83%; against **zone**, two extra blockers buy +0.6 s. *Defense cards*: for every opponent, pocket clock and race-won-allowed by coverage call (e.g. "attack their Cover-2 with quick game, they break the pocket at 2.3 s"). Per-team cards: `python team_report.py --team DAL`.

![Protection exchange rate](output/chart_protection.png)
![Team sync](output/chart_teams.png)


**What exists vs. what's new**

| Already public | This project |
|---|---|
| NGS *average separation* — how much space at the throw | **When** the space arrived, relative to the protection |
| NGS *time to throw*, PFF *pressure rate* — the pass rush alone | Pass rush and routes **on the same clock, per play** |
| PFF pass-blocking grades (paywalled) | **Exchange rate**: pocket time bought vs. receivers given up, by coverage |
| 2023 Big Data Bowl: linemen and rushers only | Receivers (WR/TE/RB) measured against that pass rush |

**Who uses it.** Broadcasters get a live TV graphic (`race_tv.py`: two clocks racing above the play, in the spirit of the IPL's required-run-rate bar) with an AI on-air call generated from the clock numbers by a local LLM (`race_commentary.py`, Qwen via Ollama, rule-based fallback); scouts get receivers who win fast rather than receivers who are merely open; coaches get a team sync number that says "call quicker routes" or "keep a blocker in."

**Run it.**
```
python compute_race.py          # ~4 min, all 122 games -> output/race_routes.csv, race_players.csv, race_teams.csv
python make_charts.py           # -> output/chart_receivers.png, chart_teams.png, chart_positions.png
python race_tv.py               # -> output/race_tv_<game>_<play>.html  (broadcast mock; open in a browser)
python race_tv.py --game 2021090900 --play 687
python race_commentary.py --game 2021090900 --play 687 --inject   # AI on-air call via local Qwen (Ollama); template fallback
python team_report.py --team BAL   # -> validation_target.csv, protection_exchange.csv, defense_cards.csv + charts
python model.py                     # -> model_results.csv, robustness.csv, chart_model.png (numpy logistic regression)
RACE_PRESSURE=1.5 python compute_race.py --games 30   # sensitivity run (also RACE_OPEN, RACE_STEM); feeds model.py
python build_site.py                # -> index.html, the play viewer with 24 embedded plays
```
Requires pandas, numpy, matplotlib (`pip install -r requirements.txt`). Optional: Ollama + `ollama create race-caller -f Modelfile` for the AI on-air call. Data folder `nfl-big-data-bowl-regional-event-data-main/data/` sits beside the scripts.

**Assumptions & limitations.** The route clock rewards quick-developing routes by design: a deep route that comes open at 2.4 s "loses" to a 2.2 s pocket even if it would have produced a 40-yard gain, so this is a measure of timing, not of total receiving value — pairing it with air yards / EPA from nflverse is the obvious next step. Matchups are inferred as the nearest coverage defender (PFF does not label assignments). Thresholds (2 yds pressure, 3 yds open, 5 yd stem, no win before 1.0 s) are football-reasonable but not tuned; the tracking stops at the throw, so the metric describes the race up to the release, not the catch. Eight weeks of data: receiver estimates use a 30-route minimum and should still be read with error bars in mind.

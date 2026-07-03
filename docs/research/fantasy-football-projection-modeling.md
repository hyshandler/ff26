# NFL Fantasy Football Predictive Modeling: Research Notes

Research conducted 2026-07-03 for a LightGBM-based NFL fantasy projection model on nflverse data
(box-score + advanced usage metrics + injury history, walk-forward backtested; team/context features
deferred to v2).

Sourcing note: primary sources (project docs, methodology pages, papers, repos) are linked directly.
Where only a secondary source could be found repeating a claim, that is flagged explicitly rather than
presented as if verified at the source.

## General approaches

**nflverse / Open Source Football (highest-trust primary source for this stack).** The nflverse
ecosystem itself does not publish an official "fantasy projection" model, but it publishes the
building blocks that every serious public model is built on. [nflreadr](https://nflreadr.nflverse.com/reference/index.html)
exposes `load_snap_counts()` (game-level snap counts from Pro Football Reference, from 2012),
`load_injuries()` (weekly injury report data), and `load_ftn_charting()` (manual charting data
including route/coverage detail from 2022 onward) as first-class, documented data sources — i.e. the
raw ingredients for target share, air yards, red-zone usage, and snap % are treated by the project
itself as core, not exotic, features. See the [nflverse data schedule](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html)
for update cadence and the participation-data caveat (pre-2023 from NFL Next Gen Stats, 2023+ from FTN).

[ffopportunity](https://github.com/ffverse/ffopportunity) (part of the ffverse, built on nflverse data)
is the clearest publicly-documented "expected fantasy points" model in this ecosystem: it uses
**XGBoost** (gradient-boosted trees, the same family as LightGBM) trained on 2006–2020 nflverse
play-by-play to predict expected outcomes (completions, yards, TDs) from situational features
(down/distance, air yards, field position, etc.), explicitly to separate "opportunity quality" from
noisy box-score outcomes. This is direct evidence that GBM-family models on nflverse play-by-play is
an established, not novel, approach for this problem.

[Open Source Football's "Calculating Expected Fantasy Points for Receivers"](https://opensourcefootball.com/posts/2020-08-30-calculating-expected-fantasy-points-for-receivers/)
(Anthony Reinhard) documents the methodology in more detail: it combines nflfastR's completion
probability (CP) and expected-yards-after-catch (xYAC) models, multiplied out across the full
distribution of possible outcomes on a given target (accounting for air yards, down/distance, yard
line, and QB quality), to compute what an *average* receiver would be expected to score on that
target. The author's own stated finding on predictive power is important: expected fantasy points are
good at identifying **unsustainable performances** (his Sammy Watkins example: a 99th-percentile
week-1 outcome followed by a 4th-percentile week 2) but the model does **not** claim to predict
week-to-week volume, which the author says "remains unpredictable due to game script and player
health."

**Target variable definition.** Public write-ups split between weekly points and season totals
depending on use case (weekly is used for start/sit and DFS, season totals for draft rankings), but
no primary source argues for predicting season totals *directly* as a single regression target for a
usage-driven model — the general pattern (visible in ffopportunity, in numberFire's stated approach,
and in the Kaggle/academic work below) is to model at the game/week grain and aggregate, because that
grain is where the volume-driving features (snaps, targets, red zone looks) actually vary and can be
attributed to real events (injuries, role changes, game script).

**FantasyPros.** Their own [in-season accuracy methodology page](https://www.fantasypros.com/about/faq/football-inseason-accuracy-methodology/)
is primary-sourced but is about *evaluating* the ~100+ contributing experts' Expert Consensus Rankings
(ECR) against realized production, not about how FantasyPros itself derives numeric projections — they
snapshot every expert's rankings before Thursday night's game and again before Sunday 1pm games, then
score each expert by comparing the historical average fantasy output at the rank slot they gave a
player against that player's actual output. **FantasyPros' own projection-construction algorithm is
not publicly documented** on their methodology pages as far as this research could locate — I could
not find a primary statement of how their numeric (points) projections, as opposed to ECR, are
computed; treat any description of that as unverified.

**PFF.** PFF does not publish a single "how our fantasy projections are built" page, but two of their
own analytical articles are directly relevant and primary:
- [PFF's target-share model piece, "Refining 'Coach, I was open'"](https://www.pff.com/news/nfl-refining-coach-i-was-open-a-model-comparing-actual-vs-predicted-target-shares)
  builds *predicted* target share and *predicted* air-yards-share models from PFF's route-level
  charting data (receiver grade, route type, separation) and reports that **predicted target share
  explains more next-week target share variance than actual target share does (48% vs. 41% R²)**, and
  similarly predicted air-yards-share beats actual air-yards-share by ~0.1 R². This is a primary,
  quantified claim that a model-derived "who *should* be getting the targets" signal outperforms the
  raw box-score "who *did* get the targets" signal for forecasting future usage.
- [PFF's "Projecting injury risk and games missed"](https://www.pff.com/news/fantasy-football-projecting-injury-risk-and-games-missed-for-the-top-players-in-best-ball)
  is directly relevant to the games-played question and is discussed in depth below — notably, PFF is
  explicit that they consider injury prediction an open problem, not a solved one.

**numberFire, PlayerProfiler, ESPN, Establish The Run.** These commercial services are much less
forthcoming about internal mechanics than FantasyPros/PFF:
- numberFire's own site states only that it uses "statistical models and expert analysis," updated for
  "injuries and other relevant factors," and mines historical results — no architecture or feature list
  is published (per [FantasyPros' description of numberFire](https://www.fantasypros.com/experts/numberfire/)
  and secondary coverage; **numberFire has not published a primary methodology page** as far as this
  research could find).
- [PlayerProfiler's own About page](https://www.playerprofiler.com/about/) states its philosophy as
  distilling "opportunity, productivity, and efficiency-related information" into predictive metrics,
  and its [glossary](https://www.playerprofiler.com/terms-glossary/) documents specific named metrics
  (Speed Score, College Dominator Rating, Breakout Age, Production Premium) but does not publish a
  unified predictive-model methodology — it is a metrics/glossary site, not a documented model.
  Notably its own "Breakout Finder" for prospects is described as analyzing "college production,
  measured athleticism, teammate scores, level of competition, age, and draft capital" — i.e. explicit
  use of draft capital as a first-class predictive feature, echoed independently below in the rookie
  WR findings.
- ESPN's [Mike Clay 2026 Fantasy Football Projection Guide PDF](https://g.espncdn.com/s/ffldraftkit/26/NFLDK2026_CS_ClayProjections2026.pdf)
  exists and is attributed directly to Clay, but the document could not be parsed as text via automated
  fetch in this session (image/binary-heavy PDF); its detailed methodology could not be verified
  first-hand here — flagging this rather than guessing at contents.
- Establish The Run states only that it uses "a proprietary model plus a unique understanding of player
  usage and team rotations" ([ETR weekly projections page](https://establishtherun.com/fantasy-point-projections/));
  no architecture is disclosed publicly.

**Academic / competition work.**
- [Method and Validation for Optimal Lineup Creation for Daily Fantasy Football Using Machine Learning
  and Linear Programming (arXiv 2309.15253)](https://arxiv.org/abs/2309.15253) uses a supervised neural
  network trained on 2018 NFL season data to project fantasy points, feeding a mixed-integer linear
  program for DFS lineup optimization. The paper's own validation compared its generated lineups
  against real DraftKings user lineups and landed around the **31st percentile** — i.e. the authors'
  own reported result is that their model underperformed the median human DFS player, a useful
  data point on how hard beating the market actually is even with ML in the loop.
- [OpenFPL (arXiv 2508.09992)](https://arxiv.org/abs/2508.09992) is Fantasy *Premier League* (soccer),
  not NFL, but is a well-documented primary methodology (position-specific ensemble models, walk-forward
  style evaluation across seasons) worth noting as a cross-sport analog for ensemble/position-specific
  model design, not as NFL-specific evidence.
- Kaggle: no dedicated, well-populated "predict NFL fantasy points" competition with a canonical
  winning methodology write-up was found via direct kaggle.com search. What exists are (a) the annual
  [NFL Big Data Bowl](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction) competitions,
  which are about play-level tracking-data prediction (e.g. player movement), not fantasy scoring, and
  (b) scattered small community datasets/notebooks (e.g. `nfl-stats-1999-2022`,
  `fantasy-football-performance`) without rigorous methodology write-ups. This is worth noting directly:
  **there is no major, well-documented Kaggle competition specifically for NFL fantasy football point
  prediction**, unlike (for example) soccer/FPL, which has more developed public competition history.

## Games-played / injury / availability prediction

This is treated by practitioners as a genuinely separate, harder problem than points-if-playing
projection, and the state of public evidence is mixed-to-skeptical on how much signal is really there.

**PFF's own framing is the clearest primary statement of the field's uncertainty.** In
["Projecting injury risk and games missed for the top players in best ball"](https://www.pff.com/news/fantasy-football-projecting-injury-risk-and-games-missed-for-the-top-players-in-best-ball),
PFF describes using Bayesian forecasting over their own historical injury dataset to estimate
weekly active/inactive probability by position, but states outright: **"we are yet to answer the
biggest questions — can we really predict in-season injuries, and if so, what inputs are most
valuable?"** They cite Football Outsiders' team-level Adjusted Games Lost metric as showing a
**0.33 year-over-year correlation** (moderate, not strong, and at the *team* level, not the
individual-player level) as the most concrete piece of quantified evidence they reference. PFF does
not claim strong player-level predictive accuracy for their own model.

**Individual injury-history-as-predictor claims are weak or contested.** Search results (via
secondary discussion, not a single strong primary source) repeatedly surface the claim that "missing
games in year N was barely a factor for missing games in year N+1" and that some analysts "would
rather ignore [injury history] in projecting players" — this could not be traced to one specific named
primary analysis in this session and should be treated as a commonly-repeated practitioner heuristic
rather than a verified statistic; flagging explicitly rather than presenting as sourced.

**Workload ("mileage") as an injury predictor is actively disputed.** [Footballguys' "The RB Mileage
Myth: What the Numbers Say About Workload and Injuries"](https://www.footballguys.com/article/2025-running-back-milage-myth-what-numbers-say-about-workload-injuries)
(Adam Hutchison) frames prior-season touch volume as popularly assumed to predict future injury, and
sets up the counter-evidence, but the full data comparison sits behind a paywall the fetch could not
access — the article's headline claim ("myth") could not be independently confirmed against its
underlying numbers in this session. Separately, a peer-reviewed clinical study,
["Increased Snap Counts Were Not Seen Prior to Lisfranc Injuries in the National Football League" (PMC9851848)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9851848/),
is a genuine primary academic source and found **no elevated snap-count pattern preceding this specific
injury type** — direct evidence against a naive "recent workload spike → injury" feature, at least for
Lisfranc injuries specifically (not generalizable to all injury types without more evidence).

**Baseline injury/availability rates are well established and position-dependent**, which is useful
context even without a strong predictive model: Draft Sharks (in
["Should Injury Risk Stop You From Drafting RBs Early?"](https://www.draftsharks.com/article/should-injury-risk-stop-you-from-picking-running-backs-early),
using their own "Sports Injury Predictor" dataset of 2,190 injuries since 2012) reports that top-12 RBs
and WRs had **nearly identical injury counts** over four seasons, with RBs missing only about one more
game on average (2.81 vs 1.85 games) — i.e. the RB-specific "injury tax" is real but smaller than
popular narrative suggests, and the article stops short of quantifying any model's actual predictive
accuracy (precision/recall, calibration) for *which* players will get hurt. Separately, secondary
commentary (not independently verified against a primary Football Outsiders/PFF number here) puts NFL
RBs' chance of playing a full season at roughly 27% from 2017–2024, i.e. most running backs miss at
least one game most years — high base-rate unavailability that a v1 heuristic needs to capture even
without a sophisticated individual-risk model.

**Academic sports-injury ML more broadly** does report real, if modest, predictive performance —
[Machine Learning for Understanding and Predicting Injuries in Football (Van Eetvelde et al.,
Sports Medicine – Open, 2022)](https://link.springer.com/article/10.1186/s40798-022-00465-4) is a
genuine peer-reviewed review covering (soccer-dominant) literature, generally finding workload metrics,
age/experience, and prior injury as recurring but individually weak predictors, with models needing
rich internal (GPS/biometric) data practitioners in fantasy football simply don't have access to. A
narrower NFL-specific student paper,
["A Machine Learning Framework for Predicting NFL Injuries Based on Preceding-Play Patterns" (NHSJS)](https://nhsjs.com/2025/a-machine-learning-framework-for-predicting-nfl-injuries-based-on-preceding-play-patterns/),
is not a peer-reviewed academic source (student research journal) and its performance numbers should
be treated as low-confidence, but it's noteworthy for its framing choice: predicting injury from
*sequences of preceding plays* rather than the injury play itself — i.e. modeling accumulated in-game
load/context, not static season-level features. This is a different feature paradigm than the
age/injury-history/position features contemplated in this project's design, and is far more granular
(play-by-play within-game) than what a weekly/season fantasy model would typically use.

**Bottom line for practitioners' own stated confidence:** every primary source found here that
actually makes injury-prediction claims (PFF, the clinical Lisfranc study, the sports-medicine review)
either explicitly disclaims strong predictive power, reports only moderate/team-level correlations, or
finds a specific popular workload theory unsupported. No primary source located in this research
claims strong, validated *individual player* games-missed prediction from age/injury-history/position/
snap-count features alone.

## Surprising/counterintuitive findings and known pitfalls

- **Process metrics beat outcome metrics for forecasting future usage, with a real quantified
  edge.** PFF's own predicted-target-share model [beats actual target share for predicting *next
  week's* target share by 7 percentage points of R² (48% vs 41%)](https://www.pff.com/news/nfl-refining-coach-i-was-open-a-model-comparing-actual-vs-predicted-target-shares).
  This is the single most concrete, quantified "surprising" finding located in this research: a
  charting-derived predicted-opportunity metric outperforms the raw realized box-score stat at
  predicting the *next* realized box-score stat.

- **WOPR (Weighted Opportunity Rating)**, created by [Josh Hermsmeyer](https://www.rotoviz.com/author/jhermsmeyer/page/2/)
  (originally at FiveThirtyEight/RotoViz), formalizes combining target share and air-yards share
  (`1.5 × target share + 0.7 × air yards share`) specifically because the combination is more
  predictive of *future* fantasy points than either component alone — this is a widely-cited primary
  formula in the fantasy analytics community, though the original FiveThirtyEight article that
  introduced it could not be directly re-fetched in this session; treating the formula and its
  described weighting rationale as attributed to Hermsmeyer per multiple community sources
  referencing him directly as originator.

- **Regression to the mean is a load-bearing, explicitly-modeled concept for veteran analysts, not
  just a background statistical fact.** [Adam Harstad's "Regression Alert: When Yards Per Target is
  Off Target" (Footballguys)](https://www.footballguys.com/article/HarstadRegression03) builds an
  explicit expected-yards-per-target formula (`2.86 + 0.367 × yards per reception`) to separate
  players whose per-target efficiency is unsustainably high (his example: Sammy Watkins) from those
  being *underrated* by raw yardage because of high, more sustainable target volume (his example:
  DeAndre Hopkins/Keenan Allen) — and explicitly predicts the "underrated volume" group will
  outscore the "overperforming efficiency" group going forward. This is the same principle as
  points 1–2 above (volume/opportunity metrics > per-play efficiency for forecasting) stated as an
  explicit, named modeling technique by a specific analyst, not just an abstract stats fact.

- **Aging curves are position-specific and the popular "cliff" ages are contested even among
  specialists.** [4for4's "Production Curves" piece](https://www.4for4.com/2025/preseason/production-curves-positional-breakouts-prime-years-and-falloffs-age)
  and [Harstad's Footballguys running-back aging work](https://www.footballguys.com/article/HarstadDiP18?article=HarstadDiP18)
  are the most detailed public treatments; per secondary summaries of this work (not independently
  re-verified line-by-line here), running back fantasy production shows **no dropoff specifically at
  age 30** on a points-per-game basis, with decline more consistently emerging around age 28–29 in some
  cohort analyses — i.e. the popular "RBs fall off a cliff at 30" heuristic doesn't match the age at
  which the data actually shows decline starting. This argues for treating age as a continuous,
  position-specific feature with a data-driven curve rather than a hardcoded threshold.

- **Draft capital is a first-class predictive feature for rookies, not just a demographic control.**
  Multiple independent sources converge on this: [PFF's rookie WR Week 1 breakout piece](https://www.pff.com/news/fantasy-football-rookie-wide-receiver-week-1-breakouts-why-they-are-predictive-of-future-fantasy-success)
  and secondary analytics roundups both report that **83.9% of rookie WR fantasy breakouts over 12
  years were players drafted in Rounds 1–2**, and that essentially no Round-5+ rookie WR has broken
  out except one notable outlier (Puka Nacua) — draft capital functions as a strong prior even before
  in-season usage data exists, which matters directly for cold-start prediction (rookies, new team
  situations) where box-score history doesn't exist yet.

- **Known modeling pitfalls flagged by the sources reviewed** (synthesizing across the injury,
  regression, and general ML-methodology sources above, since no single nflverse-specific backtesting
  guide was found): target/outcome leakage from season-level aggregate features that partially encode
  the target itself; temporal leakage from randomly splitting seasons rather than holding out
  chronologically; and look-ahead leakage from including in-season advanced-charting data (e.g. FTN
  charting, which per [nflreadr's own documentation](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html)
  is published with up to a 48-hour lag after each game, and participation data which is only released
  post-season for years after 2023) that would not actually have been available to a bettor/projector
  at true prediction time.

## Sources of edge over generic consensus/ADP

- **Situational/opportunity signals not yet priced into box scores.** The PFF predicted-target-share
  finding above is directly an "edge" claim: a model that estimates *who should be getting volume*
  from route/separation/grade data ahead of when the box score confirms it is, by PFF's own reported
  numbers, more predictive of near-term target share than the box score itself — i.e. this kind of
  signal can front-run consensus rankings that are still anchored to trailing box-score volume.

- **Rookie/early-usage signal ahead of full-season sample.** The rookie WR breakout literature above
  (draft capital + early target share/route participation + first-downs-per-route) is explicitly framed
  by its own sources as identifying ascending players "before production hits the box score" — i.e.
  the edge is in reading process metrics during a player's first handful of games, before ADP/consensus
  has caught up to a role change.

- **Regression-based fades/buys.** Harstad's yards-per-target regression framework is explicitly a
  "beat the market" tool: identify players whose *box-score output* diverges from what their
  *volume/opportunity* would predict, and expect convergence — this is a direct, named practitioner
  technique for gaining edge over rankings that are still reacting to recent box-score results.

- **Injury-risk mispricing is asserted more often than proven.** Multiple sources here (Draft Sharks,
  PFF) gesture at injury-risk-adjusted valuation as a potential edge, but as documented above, none of
  the primary sources found actually validate strong individual-player injury prediction — so
  "injury-risk edge" should be treated as a plausible-but-unproven edge source per the current public
  evidence, not a demonstrated one. This tempers optimism about injury-risk-adjusted ADP arbitrage as
  a v1 edge source.

- No primary source reviewed here made a rigorously quantified, apples-to-apples claim of "our model
  beats ADP/consensus by X%" with a transparent methodology — accuracy claims (e.g. numberFire's
  historical claim of "outpredicting ESPN and Yahoo 70% of the time," found only via secondary/NBC News
  coverage, not a numberFire primary page) should be treated as unverified marketing claims rather than
  audited results.

## Implications for our model

**1. Games-played submodel: recommend punting to a simple heuristic in v1, not building a real
submodel — the evidence base does not currently support it.** Every primary source that directly
engages with individual-player injury/availability prediction (PFF's own writeup, the Lisfranc clinical
study, the sports-medicine ML review, Draft Sharks' own injury-risk piece) either explicitly says the
field hasn't answered "can we predict in-season injuries" ([PFF](https://www.pff.com/news/fantasy-football-projecting-injury-risk-and-games-missed-for-the-top-players-in-best-ball)),
reports only moderate team-level (not player-level) correlation (Football Outsiders' 0.33 YoY AGL
correlation, as cited by PFF), or finds a specific popular workload-based theory unsupported for a
given injury type (the Lisfranc study). No source here demonstrates validated individual-player
games-missed prediction from age/injury-history/position/snap-count features with reported accuracy
numbers worth building against. Given that, a v1 heuristic — e.g. position-and-age-conditioned base
availability rate (recall RBs run roughly a ~27%-full-season / ~2.8-vs-1.9-games-missed gap vs. other
skill positions per the Draft Sharks/secondary figures above), possibly adjusted for a simple "coming
off IR/recent major injury" flag — will likely capture most of the *available* signal at a fraction of
the engineering cost, and building a full submodel risks overfitting to noise the literature itself
says isn't clearly there. Revisit for v2 only if nflverse's `load_injuries()` weekly report data (which
does exist and is a real, structured primary data source, unlike ad hoc scraped injury text) shows
measurable signal in a quick correlation check — that's a cheap validation step worth doing before
committing engineering time to a full submodel.

**2. Target variable: the evidence favors modeling at the game/week grain over season totals, then
aggregating up.** Nothing in the public methodology sources argues for predicting season totals
directly; ffopportunity, Open Source Football's xFP work, and the DFS/academic literature all operate
at the play/game level specifically because that's the grain where the causal usage features (snaps,
targets, red-zone looks, injuries, role changes) actually live and can be attributed to real events.
Given this project already has injury history and weekly snap %/target share/air-yards features in
scope, points-per-game (predicted per eligible game, then multiplied by a separate games-played
estimate) is the more defensible target than season-total regression — it also cleanly decouples the
"how good will they be when they play" question (where the box-score/process features do real work)
from the "will they play" question (where, per point 1, a v1 heuristic is the honest current state of
the art), rather than conflating both into one noisy season-total number.

**3. Yes — prioritize process/opportunity metrics (target share, air yards, snap %, red-zone usage)
over raw box-score outcome stats (receptions, TDs, yards) as model features, and this is the
best-evidenced recommendation in this entire research pass.** The PFF predicted-target-share finding
(7 points of R² advantage predicting *future* usage over the raw stat), the WOPR formula's explicit
design rationale (volume+air-yards combo beats either alone for forecasting), and Harstad's
regression-to-volume framework all converge independently on the same point: usage/opportunity
signals are more stable and more predictive of *future* box-score output than trailing box-score
output is of itself. Concretely for feature engineering: target share, air yards share, red-zone
target/carry share, and snap % should be treated as first-order predictive features (not just
descriptive ones), and where possible should be given more recency-weight / stability than raw
weekly receptions/yards/TDs, which the same sources describe as noisier and more subject to
game-script variance.

**4. Walk-forward backtesting pitfalls specific to this LightGBM + nflverse setup:**
- **Advanced-metric availability lag.** nflreadr's own documentation states FTN charting data is
  published up to 48 hours after a game and that pre-2023 participation data comes from a different
  source (NGS) than post-2023 (FTN) — a naive backtest that just joins full-season files by week
  number will silently use metrics that weren't actually available at true prediction time, and will
  also train on a discontinuity in how snap/route participation was even measured across the
  2022/2023 boundary. Any walk-forward split must respect actual publication timing, not just game
  date, and should treat the pre/post-2023 participation-data source change as a potential
  distribution shift to control for, not ignore.
- **Season-level aggregate leakage.** Because target share, air yards share, and red-zone share are
  natural candidates as *rolling* features, care is needed that any "share of team X" feature computed
  for week N only uses team totals through week N-1, not a season-end aggregate — a season-total
  denominator leaks future information (e.g. a player's own huge week 17 game inflates the "share"
  denominator available at week 3).
- **Survivorship bias from only including players who stayed on-roster/healthy.** Given point 1's
  conclusion that a v1 games-played heuristic is doing real work outside the LightGBM points model, the
  points-if-playing training set is implicitly conditioned on "played that week" — which is fine for
  that submodel's narrow question, but the *evaluation* of the combined system (points-per-game ×
  games-played) needs to be checked against season-total or full-slate outcomes to make sure the
  survivorship conditioning in the points model isn't silently making the combined system look better
  in backtest than it will in production (e.g. a player who got hurt and never returned effectively
  drops out of the "played" training distribution entirely, rather than being scored as a bust).
- **Look-ahead bias from features that are themselves partly forecasts.** Some publicly-used features
  (e.g. PFF's predicted target share, ffopportunity's expected fantasy points) are themselves the
  output of models trained on data that may span the full historical window being backtested against —
  if this project ever ingests a similarly model-derived feature (vs. raw nflverse charting), the
  provenance and training window of that upstream model needs to be walk-forward-safe too, not just
  this project's own LightGBM model.
- **Small-sample noise at the weekly grain**, flagged repeatedly across sources (Open Source Football's
  own caveat that xFP "don't predict actual performance reliably" week to week; the DFS neural-net
  paper's modest 31st-percentile real-world result) — expect meaningfully higher backtest variance at
  the single-week grain than at multi-week aggregates, and prefer reporting backtest error metrics
  (e.g. MAE) alongside a baseline (season-average, or ADP-implied projection) rather than in isolation,
  since several sources here note that even well-resourced commercial models struggle to convincingly
  beat simple baselines on a single-week basis.

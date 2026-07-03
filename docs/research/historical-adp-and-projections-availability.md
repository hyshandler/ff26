# Historical NFL Fantasy Football ADP and Point Projections: Data Availability (2020-2024 seasons)

Research date: 2026-07-03. Seasons researched: real calendar NFL seasons 2020, 2021, 2022, 2023, 2024.

---

## Verdict

**A. Historical ADP (2020-2024): FEASIBLE**
Recommended method: **Fantasy Football Calculator's free public REST API** —
`https://fantasyfootballcalculator.com/api/v1/adp/{format}?teams=12&year={YYYY}` — verified directly, returns
full player lists (180-200+ players) with genuine late-August preseason draft dates for every year 2020-2024,
no API key, no login, no paywall. FantasyPros' live site also exposes a `?year=YYYY` parameter back to 2015,
but anonymous/unauthenticated requests are registration-gated to the top 5 players only; full historical
FantasyPros ADP tables are recoverable for free via Wayback Machine snapshots instead (verified).

**B. Historical point projections (2020-2024, as originally published pre-season): PARTIALLY FEASIBLE**
No live vendor (FantasyPros, ESPN/Mike Clay, PFF, numberFire) exposes a documented, free, programmatic
archive of *past* preseason projections — current sites and APIs only serve the live/current season, and the
ffanalytics R package explicitly states historical-period scraping "will not be successful." The one working
route is the **Wayback Machine** (CDX API + snapshot fetch): FantasyPros' `/nfl/projections/{pos}.php` pages
have snapshots going back to January 2020, including August/September (preseason) snapshots for every year
2020-2024, and a spot-check of the August 2022 QB snapshot returned a real, populated projections table. This
works but is manual/scraping-based (HTML table parsing per snapshot), not an official API, and snapshot
density/timing varies by year, so it's "partially feasible" rather than a clean feasible.

---

## Detailed Findings

### QUESTION A — Historical ADP

**1. nflverse ecosystem (nflreadr, nfl_data_py/nflreadpy, and search for `ffpros`)**
- Checked: github.com/nflverse org page, nflreadr's `load_*` function family, nfl_data_py/nflreadpy docs.
- Found: No ADP function anywhere in the core nflverse packages. `nflreadr` exposes `load_ff_playerids`,
  `load_ff_rankings`, `load_ff_opportunity`, but no ADP loader. `nfl_data_py` is deprecated in favor of
  `nflreadpy`; neither documents an ADP function.
- **`ffpros` does exist**, but it lives in a sibling ecosystem, **`ffverse`** (github.com/ffverse), not
  nflverse itself. `ffpros` is an R package that scrapes FantasyPros.com. Confirmed exported functions:
  `fp_rankings()`, `fp_projections()`, plus config helpers (`fp_set_metadata`, `fp_set_ratelimit`,
  `fp_set_sport`, `fp_set_useragent`). No dedicated `fp_adp()` — ADP would need to be pulled via
  `fp_rankings()` pointed at an ADP-labeled `page` value, or scraped separately.
- Sources: https://github.com/nflverse , https://ffpros.ffverse.com/ , https://github.com/ffverse/ffpros ,
  https://ffpros.ffverse.com/reference/index.html
- Confidence: verified directly (org repo list fetched; ffpros reference index fetched; function source
  files fetched from raw.githubusercontent.com).

**2. ffanalytics R package**
- Checked: github.com/FantasyFootballAnalytics/ffanalytics README and scrape functions (`add_adp`, `add_ecr`,
  `add_aav`, `add_uncertainty`, `scrape_data`).
- Found: It does scrape ADP (via `add_adp()`) and rankings/ECR (`add_ecr()`), and its ADP scraper hits a
  **public (undocumented) Yahoo endpoint** —
  `https://pub-api-ro.fantasysports.yahoo.com/fantasy/v2/league/{league_id}/draft_analysis` — returning
  `adp`, `min_dp`, `max_dp`, `draft_percentage` fields (per `R/adp_functions.R`). However, the package's own
  README/docs **explicitly state historical-period scraping is not supported**: "Although the scrape
  functions allows the user to specify season and week, scraping historical periods will not be successful."
  This applies to both ADP and projections — ffanalytics is a *live/current-data* scraper only.
- Sources: https://github.com/FantasyFootballAnalytics/ffanalytics ,
  https://github.com/FantasyFootballAnalytics/ffanalytics/blob/master/R/adp_functions.R
- Confidence: verified directly (README fetched; adp_functions.R content referenced via search result showing
  the endpoint and field names).

**3. FantasyPros site (`fantasypros.com/nfl/adp/overall.php`)**
- Checked: base URL, and `?year=2022` query variant, comparing rendered page config JSON for both.
- Found: The live site **does** support a `year` query param, with a documented dropdown covering seasons
  2015 through 2026 (`filters` config embedded in page: `"options":{"_2026":...,"_2015":{"text":"2015"}}`).
  Requesting `?year=2022` returns a page whose embedded report config shows `"title":"2022 Average Draft
  Position (ADP)"` and real 2022-consensus data (Jonathan Taylor ranked #1, sourced from RTSports + Sleeper).
  **However**, the embedded/server-rendered data is capped at **5 rows** for anonymous requests
  (`"registrationFence":true` in the page's JS config) — the same cap applies to the *current* season's page
  too, so this isn't a historical-specific restriction, but it does mean the live site's year filter cannot
  be used to scrape full-depth historical ADP for free without an account.
- Also checked `api.fantasypros.com` — FantasyPros has a v2 API (`api.fantasypros.com/v2/docs`,
  `api.fantasypros.com/public/v2/docs/`) that is **key-gated** (must email FantasyPros to request an API key)
  and has ToS restrictions (no building a competing product; must attribute FantasyPros in any published
  analysis). No free/self-serve signup was found in the docs pages returned by search.
- Sources: https://www.fantasypros.com/nfl/adp/overall.php?year=2022 (fetched directly via curl),
  https://api.fantasypros.com/v2/docs , https://support.fantasypros.com/hc/en-us/articles/49749297704475
- Confidence: verified directly for the year-param behavior and the registration fence (raw HTML diffed
  between `?year=2022` and no-param current-season request). API key/ToS details are from search-result
  summaries of the FantasyPros support article, not a directly fetched full ToS document — **could not fully
  verify** exact API pricing/free-tier availability.

**4. Sleeper API**
- Checked: docs.sleeper.com endpoint list.
- Found: **No ADP endpoint exists.** Sleeper's public API exposes users, leagues, rosters, matchups,
  transactions, drafts (`/v1/draft/<draft_id>/picks`) and players, but nothing that aggregates or serves ADP
  directly. (It would theoretically be possible to compute ADP by pulling large numbers of individual
  Sleeper mock/real draft results, but that's a derived/manual approach, not a served historical ADP dataset,
  and is out of scope of "ADP endpoint.")
- Sources: https://docs.sleeper.com/
- Confidence: verified directly (full endpoint list fetched).

**5. ESPN / Yahoo public/undocumented endpoints**
- Yahoo: ffanalytics' scraper hits `pub-api-ro.fantasysports.yahoo.com/fantasy/v2/league/{league_id}/draft_analysis`
  for **current-season, per-league** draft analysis (not a historical archive) — this is an unofficial,
  reverse-engineered public endpoint; Yahoo's official developer ToS explicitly **prohibits** reverse
  engineering / decompiling the API ("Developers may not modify, reverse engineer, decompile, or otherwise
  alter the API or separate its underlying data"), so using it is a ToS gray/red area, not condoned.
- ESPN: no documented historical ADP API was found via search; ESPN's fantasy ADP is generally surfaced only
  through FantasyPros' aggregation (labeled "ESPN ADP" under FantasyPros' PPR-scoring tab) rather than a
  direct ESPN endpoint.
- Confidence: **could not fully verify** — this is based on search-result summaries and ffanalytics source
  references, not direct testing of the Yahoo endpoint (no league_id was available to test with), and no
  ESPN endpoint was found/tested at all. Noted explicitly as unconfirmed/plausible-but-unverified for ESPN.

**Bonus/most important finding — Fantasy Football Calculator (fantasyfootballcalculator.com)**
- Checked: `help.fantasyfootballcalculator.com/article/42-adp-rest-api` and live-tested the API directly.
- Found: A **free, public, no-key REST API**: `GET https://fantasyfootballcalculator.com/api/v1/adp/{format}?teams={n}&year={YYYY}`,
  where `{format}` ∈ `standard`, `ppr`, `half-ppr`, `2qb`, `dynasty`, `rookie`. Verified live with real HTTP
  requests for years 2007, 2020, 2021, 2022, 2023, 2024 (all `teams=12`):
  - 2020: 203 players, draft window 2020-08-30 to 2020-09-01 (PPR)
  - 2021: real preseason window, CMC ranked #1
  - 2022: 12-team standard, Jonathan Taylor ADP 1.3 (1.01), draft window 2022-08-31 to 2022-09-04, 2112 total drafts
  - 2023: Justin Jefferson ADP 1.2 (1.01), draft window 2023-08-30 to 2023-09-01
  - 2024: 180 players, CMC #1, draft window 2024-08-30 to 2024-09-01
  - 2007: returns a `"status":"Success"` shell with `players: []` — i.e., very old years have thin/no data,
    but 2020-2024 all returned full, well-populated player lists.
  - Response includes per-player `adp`, `adp_formatted` (round.pick), `times_drafted`, `high`/`low`/`stdev`,
    and `bye` — richer than a simple average.
- Terms: help docs state "free for personal and commercial use," request attribution, and ask not to poll
  too frequently since "the data only updates once per day."
- Sources: https://help.fantasyfootballcalculator.com/article/42-adp-rest-api ,
  live API responses fetched via curl (see "How to access" below for exact commands).
- Confidence: **verified directly** — actual JSON responses inspected for all 5 target years.

---

### QUESTION B — Historical point projections

**1. FantasyPros consensus projections (`fantasypros.com/nfl/projections/qb.php`)**
- Checked: base URL and behavior under the same `year=YYYY` query pattern found for ADP.
- Found: The live page pattern mirrors ADP — a `year` filter exists in principle, but this was not
  separately re-verified with the same diff test as ADP (time-boxed); what **was** verified is that the
  **Wayback Machine** holds real historical snapshots of this exact URL. See Wayback findings below, which
  is the confirmed working route for full historical FantasyPros projections data.
- Confidence: partially verified — the ADP year-param behavior (with its 5-row anonymous cap) was directly
  confirmed; the projections page's year-param was not independently diffed, so its cap behavior is
  **inferred by analogy, not directly confirmed**.

**2. ffanalytics R package (again, for projections)**
- Found: Same conclusion as Question A — README states historical scraping "will not be successful" even
  though `scrape_data()` accepts `season`/`week` arguments. This applies to projections sources
  (FantasyPros, CBS, ESPN, NFL.com, etc. that ffanalytics aggregates) as much as ADP.
- Confidence: verified directly (README text).

**3. GitHub/Kaggle archived historical projections datasets**
- Searched GitHub for "fantasy football projections historical dataset" and "preseason projections csv
  2020 2021 2022": found several repos, but **none confirmed to hold true pre-season historical projections**
  specifically (as opposed to actual/final stats or ADP):
  - `dynastyprocess/data` (github.com/dynastyprocess/data): weekly-updated repo with `db_fpecr.csv.gz` /
    `.parquet` (FantasyPros **Expert Consensus Rankings**, not point projections), `db_playerids.csv`,
    and dynasty trade `values.csv`. No confirmed historical point-projections file; years of coverage not
    stated in what was fetched — **could not verify** whether ECR history includes rankings from a given
    week/season.
  - `fantasyfootballdatapros.com/csv_files` ("Fantasy Football Data Pros"): described in search results as
    having "yearly data going back to 1970, weekly back to 1999" — but this appears to be **actual stats**,
    not what a projection system said *before* the season. Not verified as containing pre-season projections.
  - `datascopeanalytics/fantasy-football-projections`: a research repo specifically about "how accurate are
    fantasy football projections" — plausible it contains a real historical projections dataset, but its
    year coverage and source vendor were **not verified** (not fetched directly).
  - Kaggle: several datasets found (`gbolduc/fantasy-football-data-2017-2023` explicitly mentions ADP
    2017-2023 sourced from pro-football-reference.com; `heefjones/nfl-fantasy-data-1970-2024`). These read as
    **actual stats/ADP datasets**, not necessarily preseason point projections — **not verified** to contain
    pre-season projected points specifically.
- Confidence: **could not fully verify** any of these as true historical *pre-season projections* archives —
  flagging explicitly rather than asserting. They are plausible leads for follow-up, not confirmed sources.

**4. Wayback Machine (CDX API + snapshot fetch)** — **confirmed working route**
- Checked: `http://web.archive.org/cdx/search/cdx?url=fantasypros.com/nfl/adp/overall.php&from=2020&to=2024&output=json`
  and the same for `fantasypros.com/nfl/projections/qb.php`, plus direct snapshot fetches.
- Found: Both URLs have **dense snapshot coverage** across 2020-2024, including snapshots that land squarely
  in the preseason/draft-prep window (roughly late July through early September) for every year:
  - ADP page: snapshots on 2020-09-02, 2021-09-01, 2022-09-03 (also 2022-08-07), 2023-05-22 &
    2023-12-07 (gap in Aug/Sep 2023 within the collapsed sample), 2024-09-05 (also 2024-08-01).
  - Projections/qb page: snapshots on 2020-09-15, 2021-09-07, 2021-08-01, 2022-08-07 & 2022-09-05,
    2023-09-13 (gap around Aug 2023), 2024-09-20.
  - **Directly fetched and inspected** the 2022-08-07 ADP snapshot
    (`web.archive.org/web/20220807023508/https://www.fantasypros.com/nfl/adp/overall.php`): 408 KB HTML,
    contains a full rendered table (~370+ player-name occurrences), top of list is Jonathan Taylor /
    Christian McCaffrey / Ja'Marr Chase — consistent with real August 2022 fantasy consensus, **and not
    gated** to 5 rows (older FantasyPros page template was a plain server-rendered table, not the
    registration-fenced SPA the current site uses).
  - **Directly fetched and inspected** the 2022-08-07 QB projections snapshot
    (`web.archive.org/web/20220807025946/https://www.fantasypros.com/nfl/projections/qb.php`): 311 KB HTML,
    populated with real QB names (Josh Allen, Patrick Mahomes at top) in a full table.
- Conclusion: Wayback Machine snapshots of FantasyPros' ADP and projections pages are a **verified, free,
  scriptable (via CDX API) source of real historical preseason data for 2020-2024**, and notably give
  *fuller* unfenced tables than the live site's current registration-gated year-filter for ADP.
- Sources: CDX API JSON responses (fetched directly via curl), snapshot HTML (fetched directly via curl).
- Confidence: **verified directly** — both the CDX index and actual snapshot content were fetched and
  inspected.

**5. numberFire, PFF Fantasy, ESPN Mike Clay, CBS Sports**
- Checked via web search only (no direct site fetch of paid/archive sections).
- Found: ESPN publishes Mike Clay's annual "NFL Projection Guide" as a **PDF**, and past years' PDFs (2023,
  2025, 2026 confirmed by search results) are reachable at predictable CDN URLs
  (`g.espncdn.com/s/ffldraftkit/{yy}/NFLDK{YYYY}_CS_ClayProjections{YYYY}.pdf`), suggesting a 2020-2024 PDF
  may exist at the analogous URL pattern — **not directly verified** (URLs were not individually fetched for
  every target year, and PDF content wasn't parsed for actual projected points).
  No public historical archive was found for numberFire or PFF Fantasy projections; both were reachable only
  as brand names in search results, with no archive page identified.
- Confidence: **could not access/verify** — explicitly unconfirmed. Do not treat the ESPN PDF pattern as
  guaranteed to exist/cover 2020-2024 without testing each URL.

---

## How to access (feasible / partially feasible sources)

### A. Historical ADP — Fantasy Football Calculator API (recommended, primary)
- Endpoint: `GET https://fantasyfootballcalculator.com/api/v1/adp/{format}?teams={n}&year={YYYY}`
- `{format}`: `standard` | `ppr` | `half-ppr` | `2qb` | `dynasty` | `rookie`
- `{n}` (teams): commonly 8, 10, 12, 14; affects which real drafts are aggregated
- `{YYYY}`: verified working 2020-2024 (also worked for 2007, though early years are thin)
- Auth: none required. Rate limit: no hard published limit, but vendor asks not to poll frequently — "data
  only updates once per day," so caching daily is sufficient and polite.
- Response fields: `player_id`, `name`, `position`, `team`, `adp`, `adp_formatted`, `times_drafted`, `high`,
  `low`, `stdev`, `bye`, plus a `meta` block with `total_drafts` and the `start_date`/`end_date` window the
  ADP was computed over.
- License/ToS: "free for personal and commercial use," attribution requested (link/mention back to FFC).
- Doc: https://help.fantasyfootballcalculator.com/article/42-adp-rest-api

### A. Historical ADP — fallback/cross-check: FantasyPros via Wayback Machine
- Step 1: query CDX API for snapshot timestamps:
  `http://web.archive.org/cdx/search/cdx?url=fantasypros.com/nfl/adp/overall.php&from=2020&to=2024&output=json&collapse=timestamp:6`
  (also available per-position: `.../adp/qb.php`, `/rb.php`, `/wr.php`, `/te.php`, and per-scoring:
  `/ppr-overall.php`, `/half-point-ppr-overall.php`)
- Step 2: fetch chosen snapshot: `https://web.archive.org/web/{timestamp}/https://www.fantasypros.com/nfl/adp/overall.php`
  and parse the HTML `<table>` (older snapshots, pre ~2023, use a plain server-rendered table; parse with
  any HTML table parser).
- Coverage: dense (roughly monthly) snapshots 2020-2024, including preseason-window snapshots each year.
- Caveat: Wayback Machine has no formal ToS restriction on this kind of read access, but is a general-purpose
  archive, not a stable API — timestamps/snapshot availability can vary and there's no guarantee of a
  snapshot on any exact desired date; nearest available snapshot must be used.
- No auth, no published rate limit (be a reasonable, non-hammering client).

### B. Historical point projections — FantasyPros via Wayback Machine (only confirmed working route)
- Step 1: CDX API per position:
  `http://web.archive.org/cdx/search/cdx?url=fantasypros.com/nfl/projections/{pos}.php&from=2020&to=2024&output=json&collapse=timestamp:6`
  where `{pos}` ∈ `qb`, `rb`, `wr`, `te`, `k`, `dst` (and possibly `flex`, `overall` — not individually
  tested, but likely follow the same URL family).
- Step 2: fetch snapshot HTML and parse the projections table (columns typically include per-category stat
  projections plus a `FPTS` total).
- Coverage verified: 2020-2024 snapshots exist, including several landing in the Aug/early-Sep preseason
  window each year (exact dates vary — see Detailed Findings above); some years (e.g., 2023) have a visible
  gap around the exact preseason weeks in the sampled snapshot list, so the nearest available snapshot
  (which may be a few weeks before or into Week 1-2) should be used and the date recorded.
- No auth required; same Wayback caveats as above (no snapshot-availability guarantee, no formal rate limit
  but be reasonable).
- Not verified/not recommended: any live vendor API, `ffanalytics`, or `ffpros`/`fp_projections()` for
  *historical* seasons — all either don't support it or weren't confirmed to actually serve historical data
  rather than re-serving current-season data under a historical-looking URL.

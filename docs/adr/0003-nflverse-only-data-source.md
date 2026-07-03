# nflverse as the sole v1 data source

Considered Pro Football Reference scraping and paid APIs (SportsDataIO, Sportradar). Chose nflverse (via `nfl_data_py`/`nflreadpy`) exclusively — free, no ToS/scraping risk, and already includes the advanced metrics (target share, air yards, snap counts) that matter most for the feature set, so there's no near-term need to integrate a second source. Adding a paid or scraped source later is a real engineering cost, not a config change, so this is a meaningful lock-in for v1.

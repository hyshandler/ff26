import re

import pandas as pd
import requests

FFC_ADP_URL = "https://fantasyfootballcalculator.com/api/v1/adp/{format}"

_NAME_SUFFIXES = re.compile(r"\s+(jr\.?|sr\.?|i{2,3}|iv|v)$", re.IGNORECASE)


def _normalize_name(name: str) -> str:
    """Strip suffixes (Jr./Sr./II/III/IV) and periods so ADP and roster names line up.

    ADP Benchmark per ADR-0007: Fantasy Football Calculator has no shared player ID
    with nflverse, so names are the only join key -- this closes most of the gap a
    naive exact match leaves (confirmed against real 2022 data: 86% exact-match
    coverage, nearly all remaining misses were suffix/punctuation formatting).
    """
    normalized = name.strip().lower().replace(".", "")
    normalized = _NAME_SUFFIXES.sub("", normalized)
    return normalized.strip()


def load_adp(year: int, teams: int = 12, adp_format: str = "ppr") -> pd.DataFrame:
    """Historical ADP for `year` from Fantasy Football Calculator's public API."""
    response = requests.get(
        FFC_ADP_URL.format(format=adp_format),
        params={"teams": teams, "year": year},
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    players = response.json()["players"]
    return pd.DataFrame(players)[["name", "position", "team", "adp"]]


def crosswalk_adp_to_player_ids(
    adp: pd.DataFrame, rosters: pd.DataFrame, season: int
) -> pd.DataFrame:
    """Adds a `player_id` column to `adp`, matched via normalized name + position
    against `rosters`' `season` rows. Unmatched rows (e.g. team defenses, or a name
    variant the normalization doesn't cover) get NaN rather than being dropped, so
    callers can see and measure match coverage."""
    season_rosters = rosters.loc[rosters["season"] == season].copy()
    season_rosters["_key"] = (
        season_rosters["player_name"].fillna("").map(_normalize_name) + "|" + season_rosters["position"]
    )
    player_id_by_key = season_rosters.drop_duplicates("_key").set_index("_key")["player_id"]

    result = adp.copy()
    result["_key"] = result["name"].map(_normalize_name) + "|" + result["position"]
    result["player_id"] = result["_key"].map(player_id_by_key)
    return result.drop(columns="_key")

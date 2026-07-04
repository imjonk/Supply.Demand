"""
Canonical watchlist generation API.

All live watchlists, historical watchlist snapshots, and future replay
snapshot consumers should import watchlist generation through this module.

For now this wraps the existing canonical implementation in watchlist.py.
Later commits can move the implementation here without changing callers.
"""

from watchlist import (
    build_watchlist,
    build_watchlist_from_zone_snapshot,
    _active_zones_for_watchlist,
    _filter_final_report,
    merge_overlapping_zones,
)

__all__ = [
    "build_watchlist",
    "build_watchlist_from_zone_snapshot",
    "_active_zones_for_watchlist",
    "_filter_final_report",
    "merge_overlapping_zones",
]
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"

WATCHLIST = [
    # Core mentor-style names
    "NVDA", "AMD", "AVGO", "TSLA", "META", "AMZN", "MSFT", "PLTR",
    "AAPL", "GOOGL", "NFLX", "COST", "INTC",

    # High-liquidity growth / semis / software
    "MU", "CRM", "PANW", "CRWD", "SNOW", "SHOP", "UBER", "ARM", "ANET", "SMCI",

    # Momentum / options favorites
    "HOOD", "SOFI", "COIN", "RBLX", "NET", "RKLB", "ASTS", "HIMS", "APP", "IONQ", "TEM", "MSTR", "NOW",

    # Market context
    "QQQ", "SPY",
]

# With 5M source bars, the scanner can build true 90m, 1H, 2H, 3H, and 4H candles
# and can also use 5M candles for more realistic in-trade exits.
# Resampling is anchored to the regular-session open (9:30 AM New York time) in data_loader.py.
SOURCE_SUFFIX_PRIORITY = ["5M", "30M", "1H"]
TIMEFRAMES = {
    "90m": "90min",
    "1H": "1h",
    "2H": "2h",
    "3H": "3h",
    "4H": "4h",
    "1D": "1D",
}

# Watchlist de-duplication / confluence merging.
# When several zones from different higher timeframes overlap, show them as one
# watchlist setup instead of separate duplicate cards. Raw zones are still saved
# to reports/detected_zones.csv.
MERGE_OVERLAPPING_ZONES = True
ZONE_MERGE_TOLERANCE_PCT = 0.25  # also merge zones separated by <= 0.25% of price/midpoint
MAX_MERGED_ZONE_WIDTH_PCT = 3.00  # avoid merging into an overly wide, unusable zone

@dataclass
class ZoneRules:
    # Base-candle filter. v0.4 is intentionally looser than v0.3 so we do not reject
    # mentor-style zones too early. Scoring/ranking should filter quality later.
    base_range_max_vs_avg: float = 1.25
    base_body_max_of_range: float = 0.65
    volume_min_vs_avg: float = 0.35
    volume_max_vs_avg: float = 2.00

    # Departure validation rule:
    # The candle immediately after the base must have a BODY greater than this
    # many times the basing candle BODY in the departure direction.
    # This uses body-to-body comparison, not full candle range or wick movement.
    use_next_candle_multiplier: bool = True
    next_candle_move_multiple: float = 2.00

    # Keep these for prior-move classification and fallback logic.
    departure_window: int = 1
    prior_window: int = 2
    min_prior_move_atr: float = 0.35
    min_departure_move_atr: float = 0.80
    # Departure volume is scored, not used as a hard zone-detection rejection.
    min_departure_volume_vs_avg: float = 0.00
    lookback_for_averages: int = 20
    max_tests_before_weak: int = 2

RULES = ZoneRules()


# Final report filtering. Keep detection broad, but only show high-quality setups in
# reports/watchlist.md and reports/watchlist.csv. Lower-grade candidates are still
# saved to reports/watchlist_all_candidates.csv for review/tuning.
FINAL_REPORT_MIN_GRADE = "A"  # Shows only A+ and A setups by default.
MAX_SETUPS_PER_SECTION = 8     # Prevents the final report from getting noisy.

# Final report R:R filter. Mentor-style setups should have enough room to
# justify the risk. The final watchlist hides any setup below 1:2.75 R:R.
# Lower R:R candidates are still preserved in watchlist_all_candidates.csv for tuning.
MIN_FINAL_RR = 2.50

# Require 5-minute source data for current scanner/backtest workflow.
# This prevents the scripts from silently falling back to old 1H files and producing
# reports that look current but are not using the intended candle structure.
REQUIRE_5M_SOURCE_FOR_WATCHLIST = True
REQUIRE_5M_SOURCE_FOR_BACKTEST = True

# Final watchlist duplicate cleanup.
# After zone merging and final grade/R:R filters, this removes duplicate trade cards
# that still point to essentially the same symbol + side + overlapping price zone.
STRICT_DEDUPLICATE_FINAL_WATCHLIST = True
FINAL_DEDUPE_TOLERANCE_PCT = 0.75


# -----------------------------------------------------------------------------
# v0.30 watchlist reset settings
# -----------------------------------------------------------------------------
# This build intentionally returns to a watchlist-first architecture. Backtesting
# remains present as a secondary utility, but watchlist inclusion is not driven by
# historical-event backtest filters.

# Target-selection quality rules for watchlist target ladders.
TARGET_ZONE_MAX_TESTS = 3
TARGET_ZONE_MIN_QUALITY_SCORE = 6.0
TARGET_SELECTION_MIN_RR = 2.50

# Watchlist zone eligibility / diagnostic rules. Detection remains broad; these
# rules mainly decide final vs developing vs research buckets.
MAX_FINAL_ZONE_TESTS = 2
MAX_RESEARCH_ZONE_TESTS = 3
ELIMINATE_ZONE_TESTS_AT = 4
ALLOW_TWO_TEST_ZONES_WITH_CONFLUENCE = True
TWO_TEST_MIN_CONFLUENCE = 2
TWO_TEST_MIN_QUALITY_SCORE = 8.0

# R:R display and ladder settings.
PREFERRED_RR_MIN = 2.50
MAX_MODELED_TARGET_RR = 6.00
TARGET_LADDER_MAX_LEVELS = 5

# v0.32 zone-accuracy settings. Watchlist/merge/target calculations use active
# zones only. Broken zones are preserved in detected_zones.csv for audit, but
# excluded from merged_zones, scenario generation, target ladders, and watchlist
# scoring.
EXCLUDE_BROKEN_ZONES_FROM_WATCHLIST_CALCULATIONS = True
TARGET_LADDER_SHOW_SOFT_OBSTACLES = True
TARGET_LADDER_PRIMARY_IS_NEAREST_OPPOSING_ZONE = True

# Scenario-prep watchlist settings. Final setups are the strictest bucket;
# developing scenarios and zone map rows are preserved for prep/context.
WATCHLIST_SCENARIO_MAX_DISTANCE_PCT = 5.0
WATCHLIST_DEVELOPING_MAX_DISTANCE_PCT = 5.0
WATCHLIST_READY_DISTANCE_PCT = 0.75
WATCHLIST_NEEDS_CONFIRMATION_DISTANCE_PCT = 2.0
WATCHLIST_INCLUDE_DEVELOPING_IN_HTML = True
WATCHLIST_INCLUDE_ZONE_MAP_IN_HTML = True
WATCHLIST_MIN_DEVELOPING_GRADE_RANK = 2  # B or better

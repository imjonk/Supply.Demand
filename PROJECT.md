# Supply/Demand Scanner + Backtester Project Rules

## Core architecture

Mission

Develop and validate a repeatable supply & demand trading process that produces an actionable pre-market watchlist with a long-term objective of approximately 60% win rate while maintaining an average 3:1 or better realized reward:risk.

Every change to the system must improve one or more parts of the trading process and be validated through historical replay against a fixed baseline before being accepted.

The watchlist is the source of truth.

Backtesting must simulate the real workflow:

1. Build RTH-only zones using data available through the prior market close.
2. Build a frozen preopen watchlist snapshot for the test date.
3. Replay that date using only candidates from the frozen snapshot.
4. Never dynamically create new trade candidates during replay.
5. Every trade must trace back to a snapshot_candidate_id.

## Data rules

- Zone creation: regular trading hours only.
- Current watchlist price: latest available market price, regardless of RTH, premarket, aftermarket, holiday, or weekend.
- Extended-hours data can inform current price/proximity/gap context.
- Extended-hours data must not create or modify zones.

## Watchlist rules

The watchlist is for preparation, not active trades.

Include only:
- price outside the candidate zone
- moving/positioned toward a mapped zone
- setup not already resolved

Hard exclude:
- current price inside candidate zone
- already rejected from candidate zone
- already broken through candidate zone
- candidates with insufficient realistic reward

Record exclusion reasons when possible.

## Backtest entry rules

A zone test is not a trade.

For rejection:
- price enters zone
- price exits the zone in rejection direction
- confirmation occurs after exit
- then entry is allowed

For continuation:
- price fully breaks through/exits zone
- confirmation occurs after full break
- then entry is allowed

## Risk/target rules

- Structural risk is based on zone height / zone invalidation.
- Main target model should compare:
  - 1R
  - 2R
  - 3R structural
  - 1x ATR
  - EMA protection exit

## EMA exit rules

Evaluate continuously after entry.

Calls/longs:
- exit after 2 closes below 9EMA

Puts/shorts:
- exit after 2 closes above 9EMA

Do not gate EMA exits behind profit milestones.

## Current priorities

v0.38 goal:
- frozen daily watchlist snapshot engine
- replay consumes only snapshots
- watchlist excludes inside-zone/resolved candidates
- candidate lifecycle/funnel analytics
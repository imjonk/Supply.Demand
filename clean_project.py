from __future__ import annotations

import argparse
import datetime as dt
import shutil
from pathlib import Path

KEEP_ROOT_FILES = {
    'README.md', 'requirements.txt', 'config.py', 'data_loader.py', 'zone_detector.py',
    'watchlist.py', 'generate_watchlist_zone_map.py', 'download_alpaca_bars.py',
    'build_backtest_snapshots.py', 'replay_backtest.py', 'analyze_backtest.py',
    'audit_exit_paths.py', 'clean_project.py',
}

KEEP_DIRS = {'data', 'reports', 'docs'}

DEFUNCT_EXACT = {
    'README (1).md', 'analyze_backtest(backup).py', 'watchlist_pre_v0366_backup.py',
    'backtest.py', 'CLEAN_BUILD_MANIFEST.md', 'PROJECT_STRUCTURE_v0_38.md',
    'performance_summary.html', 'strategy_dashboard.html', 'trades.csv', 'summary.csv',
    'snapshot_manifest.csv', 'current_prices_example.csv',
}

DEFUNCT_GLOBS = [
    '*.py.bak', '*.bak', '*.tmp', '*_backup.py', '*.py.v*_backup',
    '20??-??-??_active_zones.csv', '20??-??-??_merged_zones.csv',
    '20??-??-??_scenarios.csv', '20??-??-??_final_watchlist.csv',
    'active_zones.csv', 'detected_zones.csv', 'detected_zones.md', 'merged_zones.csv',
    'watchlist.csv', 'watchlist.html', 'watchlist.md', 'watchlist_all_candidates.csv',
    'watchlist_rejections.html', 'watchlist_rejections_*.html', 'watchlist_*.csv',
    'scenario_watchlist.csv', 'zone_map.csv',
    'confirmation_components.csv', 'entry_candidates.csv', 'entry_funnel.csv',
    'exit_reason_summary.csv', 'opportunity_cost_proxy.csv', 'performance_by_*.csv',
    'rejection_summary.csv', 'target_progress.csv',
    'exit_path_*.csv',
]

CACHE_DIRS = {'__pycache__', '.pytest_cache', '.mypy_cache'}


def collect_candidates(root: Path, include_reports_archive: bool = True) -> list[Path]:
    candidates: set[Path] = set()

    for name in DEFUNCT_EXACT:
        p = root / name
        if p.exists():
            candidates.add(p)

    for pattern in DEFUNCT_GLOBS:
        for p in root.glob(pattern):
            if p.exists():
                candidates.add(p)

    for dname in CACHE_DIRS:
        for p in root.rglob(dname):
            if p.exists() and p.is_dir():
                candidates.add(p)

    if include_reports_archive:
        for p in [root / 'archive', root / 'reports' / 'archive']:
            if p.exists():
                candidates.add(p)

    # Never touch source files, data, docs, or current reports/backtest outputs by default.
    safe = []
    for p in candidates:
        rel = p.relative_to(root)
        if rel.parts[0] in {'data', 'docs'}:
            continue
        if rel.parts[0] == 'reports' and (len(rel.parts) < 2 or rel.parts[1] != 'archive'):
            continue
        if p.name in KEEP_ROOT_FILES:
            continue
        safe.append(p)
    return sorted(safe, key=lambda x: str(x).lower())


def main():
    parser = argparse.ArgumentParser(description='Clean legacy/duplicate files from the scanner project root.')
    parser.add_argument('--apply', action='store_true', help='Actually clean. Without this, dry-run only.')
    parser.add_argument('--delete', action='store_true', help='Delete instead of archive. Default archives.')
    args = parser.parse_args()

    root = Path.cwd()
    candidates = collect_candidates(root)

    if not args.apply:
        print('DRY RUN — no files moved/deleted. Run with --apply to clean.')
        for p in candidates:
            print(f'would clean: {p.relative_to(root)}')
        return

    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_dir = root / 'archive' / f'cleanup_{stamp}'
    moved = []
    deleted = []
    if not args.delete:
        archive_dir.mkdir(parents=True, exist_ok=True)

    for p in candidates:
        if not p.exists():
            continue
        rel = p.relative_to(root)
        if args.delete:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            deleted.append(str(rel))
        else:
            dst = archive_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dst))
            moved.append((str(rel), str(dst.relative_to(root))))

    print(f'Cleaned {len(moved) + len(deleted)} item(s).')
    if moved:
        print(f'Archived to {archive_dir.relative_to(root)}')


if __name__ == '__main__':
    main()

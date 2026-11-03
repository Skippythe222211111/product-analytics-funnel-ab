from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analytics_project.analysis import (  # noqa: E402
    analyze_all,
    verify_sql_parity,
    write_results,
)
from analytics_project.simulation import DEFAULT_N_USERS, DEFAULT_SEED, generate_synthetic_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Генерация симуляции и анализ проекта.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "generated")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-users", type=int, default=DEFAULT_N_USERS)
    parser.add_argument(
        "--reuse-data",
        action="store_true",
        help="Использовать имеющиеся CSV вместо повторной генерации.",
    )
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_data:
        assignments = pd.read_csv(args.data_dir / "assignments.csv")
        events = pd.read_csv(args.data_dir / "events.csv")
    else:
        assignments, events = generate_synthetic_data(args.n_users, args.seed)
        assignments.to_csv(args.data_dir / "assignments.csv", index=False)
        events.to_csv(args.data_dir / "events.csv", index=False)

    results = analyze_all(assignments, events, seed=args.seed)
    results["sql_parity"] = verify_sql_parity(assignments, events, PROJECT_ROOT)
    write_results(
        results,
        args.report_dir / "results.json",
        args.report_dir / "results.md",
    )
    print(f"Пользователей: {results['simulation']['n_users']}")
    print(f"Событий: {results['simulation']['n_events']}")
    print(f"Сверка SQL/Python: {results['sql_parity']}")
    print(f"Результаты: {args.report_dir / 'results.md'}")


if __name__ == "__main__":
    main()

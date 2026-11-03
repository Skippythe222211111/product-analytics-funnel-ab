from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analytics_project.analysis import (  # noqa: E402
    analyze_experiment,
    analyze_funnel,
    analyze_retention,
    verify_sql_parity,
)
from analytics_project.simulation import (  # noqa: E402
    ALPHA,
    BASELINE_PURCHASE_RATE,
    MDE_ABSOLUTE,
    REQUIRED_N_PER_ARM,
    TARGET_POWER,
    estimated_power,
    generate_synthetic_data,
    required_sample_size_per_arm,
)


def epoch(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


def assignment_rows(users: list[str], complete_day: int | None = None) -> pd.DataFrame:
    assigned = epoch("2026-01-01T00:00:00Z")
    if complete_day is None:
        complete_day = epoch("2026-01-21T00:00:00Z") // 86_400
    return pd.DataFrame(
        {
            "experiment_id": "test",
            "user_id": users,
            "variant": ["control" if index % 2 == 0 else "treatment" for index in range(len(users))],
            "assigned_at": "2026-01-01T00:00:00Z",
            "assigned_epoch_s": assigned,
            "assigned_day_utc": assigned // 86_400,
            "data_complete_through_day_utc": complete_day,
        }
    )


def event_row(
    user_id: str,
    event_name: str,
    timestamp: str,
    event_id: str,
) -> dict[str, object]:
    event_epoch = epoch(timestamp)
    return {
        "event_id": event_id,
        "experiment_id": "test",
        "user_id": user_id,
        "event_name": event_name,
        "event_ts": timestamp,
        "event_epoch_s": event_epoch,
        "event_day_utc": event_epoch // 86_400,
    }


class PowerPlanTests(unittest.TestCase):
    def test_power_plan_meets_target_and_is_minimal(self) -> None:
        n = required_sample_size_per_arm()
        self.assertEqual(n, REQUIRED_N_PER_ARM)
        self.assertGreaterEqual(
            estimated_power(n), TARGET_POWER
        )
        self.assertLess(estimated_power(n - 1), TARGET_POWER)

    def test_invalid_power_parameters_raise(self) -> None:
        with self.assertRaises(ValueError):
            required_sample_size_per_arm(p_control=0.99, absolute_mde=0.02)
        with self.assertRaises(ValueError):
            required_sample_size_per_arm(alpha=0)


class FunnelBoundaryTests(unittest.TestCase):
    def test_sequential_order_duplicate_users_and_inclusive_window(self) -> None:
        assignments = assignment_rows(["u1", "u2", "u3", "u4"])
        start = epoch("2026-01-01T00:00:00Z")
        events = [
            event_row("u1", "landing_view", "2026-01-01T00:00:00Z", "u1-l1"),
            event_row("u1", "landing_view", "2026-01-01T00:00:00Z", "u1-l2"),
            event_row("u1", "signup", "2026-01-07T23:59:58Z", "u1-s"),
            event_row("u1", "activation", "2026-01-07T23:59:59Z", "u1-a"),
            event_row("u1", "purchase", "2026-01-08T00:00:00Z", "u1-p"),
            event_row("u2", "landing_view", "2026-01-01T00:00:00Z", "u2-l"),
            event_row("u2", "signup", "2026-01-01T00:00:01Z", "u2-s"),
            event_row("u2", "activation", "2026-01-01T00:00:02Z", "u2-a"),
            event_row("u2", "purchase", "2026-01-08T00:00:01Z", "u2-p"),
            event_row("u3", "signup", "2026-01-01T00:00:05Z", "u3-s-before"),
            event_row("u3", "landing_view", "2026-01-01T00:00:10Z", "u3-l"),
            event_row("u3", "signup", "2026-01-01T00:00:11Z", "u3-s-after"),
            event_row("u3", "activation", "2026-01-01T00:00:12Z", "u3-a"),
            event_row("u3", "purchase", "2026-01-01T00:00:13Z", "u3-p"),
            event_row("u4", "landing_view", "2026-01-01T00:00:00Z", "u4-l"),
            event_row("u4", "signup", "2026-01-01T00:00:00Z", "u4-s-same-time"),
        ]
        result = analyze_funnel(assignments, pd.DataFrame(events))
        self.assertEqual(int(result["n_assigned"].sum()), 4)
        self.assertEqual(int(result["n_landing_view"].sum()), 4)
        self.assertEqual(int(result["n_signup"].sum()), 3)
        self.assertEqual(int(result["n_activation"].sum()), 3)
        self.assertEqual(int(result["n_purchase"].sum()), 2)
        self.assertEqual(start + 7 * 86_400, epoch("2026-01-08T00:00:00Z"))
        parity = verify_sql_parity(assignments, pd.DataFrame(events), PROJECT_ROOT)
        self.assertIn("PostgreSQL-совместимый текст", parity["funnel"])

    def test_unassigned_user_event_is_rejected(self) -> None:
        assignments = assignment_rows(["u1"])
        events = pd.DataFrame(
            [event_row("outside", "landing_view", "2026-01-01T00:00:00Z", "outside-1")]
        )
        with self.assertRaisesRegex(ValueError, "без назначения"):
            analyze_funnel(assignments, events)

    def test_assignment_duplicate_and_invalid_variant_are_rejected(self) -> None:
        assignments = assignment_rows(["u1", "u2"])
        events = pd.DataFrame(
            [event_row("u1", "landing_view", "2026-01-01T00:00:00Z", "u1-l")]
        )
        duplicate = pd.concat([assignments, assignments.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "только одно назначение"):
            analyze_funnel(duplicate, events)
        assignments.loc[0, "variant"] = "other"
        with self.assertRaisesRegex(ValueError, "variant"):
            analyze_funnel(assignments, events)


class RetentionTests(unittest.TestCase):
    def test_calendar_day_retention_deduplicates_activity_and_excludes_immature(self) -> None:
        assignments = assignment_rows(["u1", "u2", "u3"], complete_day=epoch("2026-01-08T00:00:00Z") // 86_400)
        events = [
            event_row("u1", "landing_view", "2026-01-01T00:00:00Z", "u1-l"),
            event_row("u1", "signup", "2026-01-01T23:30:00Z", "u1-s"),
            event_row("u1", "active", "2026-01-02T00:05:00Z", "u1-d1"),
            event_row("u1", "active", "2026-01-02T10:00:00Z", "u1-d1-dup"),
            event_row("u1", "active", "2026-01-08T09:00:00Z", "u1-d7"),
            event_row("u1", "active", "2026-01-03T09:00:00Z", "u1-d2"),
            event_row("u2", "landing_view", "2026-01-01T00:00:00Z", "u2-l"),
            event_row("u2", "signup", "2026-01-01T12:00:00Z", "u2-s"),
            event_row("u3", "landing_view", "2026-01-02T00:00:00Z", "u3-l"),
            event_row("u3", "signup", "2026-01-02T12:00:00Z", "u3-s"),
        ]
        result = analyze_retention(assignments, pd.DataFrame(events))
        self.assertEqual(int(result["cohort_users"].sum()), 2)
        self.assertEqual(int(result["d1_users"].sum()), 1)
        self.assertEqual(int(result["d7_users"].sum()), 1)
        self.assertEqual(len(result), 2)
        verify_sql_parity(assignments, pd.DataFrame(events), PROJECT_ROOT)


class ExperimentAnalysisTests(unittest.TestCase):
    def test_two_sided_test_interval_and_srm_flag(self) -> None:
        balanced = assignment_rows([f"b{index}" for index in range(2_000)])
        balanced["variant"] = ["control"] * 1_000 + ["treatment"] * 1_000
        counts = pd.DataFrame(
            [
                {"experiment_id": "test", "variant": "control", "n_purchase": 120},
                {"experiment_id": "test", "variant": "treatment", "n_purchase": 160},
            ]
        )
        result = analyze_experiment(balanced, counts)
        self.assertTrue(result["statistically_significant"])
        self.assertEqual(len(result["ci_95_newcombe"]), 2)
        self.assertGreater(result["ci_95_newcombe"][0], 0)
        self.assertFalse(result["ci_lower_at_least_mde"])
        self.assertFalse(result["srm_detected"])

        unbalanced = assignment_rows([f"u{index}" for index in range(2_000)])
        unbalanced["variant"] = ["control"] * 1_300 + ["treatment"] * 700
        srm_result = analyze_experiment(unbalanced, counts)
        self.assertTrue(srm_result["srm_detected"])


class SqlParityTests(unittest.TestCase):
    def test_generated_data_matches_sql_and_python(self) -> None:
        assignments, events = generate_synthetic_data()
        parity = verify_sql_parity(assignments, events, PROJECT_ROOT)
        self.assertIn("PostgreSQL-совместимый текст", parity["funnel"])
        self.assertIn("PostgreSQL-совместимый текст", parity["retention"])


if __name__ == "__main__":
    unittest.main()

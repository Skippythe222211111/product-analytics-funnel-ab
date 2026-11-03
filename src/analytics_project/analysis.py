"""Расчёт воронки, удержания и заранее заданного A/B-теста."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from .simulation import (
    ALPHA,
    BASELINE_PURCHASE_RATE,
    DEFAULT_SEED,
    MDE_ABSOLUTE,
    REQUIRED_N_PER_ARM,
    SRM_ALPHA,
    TARGET_POWER,
    estimated_power,
    required_sample_size_per_arm,
)


FUNNEL_STAGES = ("landing_view", "signup", "activation", "purchase")
WINDOW_SECONDS = 7 * 24 * 60 * 60


def validate_inputs(assignments: pd.DataFrame, events: pd.DataFrame) -> None:
    required_assignments = {
        "experiment_id",
        "user_id",
        "variant",
        "assigned_epoch_s",
        "data_complete_through_day_utc",
    }
    required_events = {
        "event_id",
        "experiment_id",
        "user_id",
        "event_name",
        "event_epoch_s",
        "event_day_utc",
    }
    missing_a = required_assignments - set(assignments.columns)
    missing_e = required_events - set(events.columns)
    if missing_a or missing_e:
        raise ValueError(f"Нет обязательных колонок: assignments={missing_a}, events={missing_e}")
    if assignments.duplicated(["experiment_id", "user_id"]).any():
        raise ValueError("Пользователь может иметь только одно назначение в эксперименте.")
    if not bool(assignments["variant"].isin(["control", "treatment"]).all()):
        raise ValueError("variant должен быть control или treatment.")
    if events["event_id"].duplicated().any():
        raise ValueError("event_id должен быть уникальным.")
    assigned_users = set(
        zip(assignments["experiment_id"], assignments["user_id"], strict=True)
    )
    event_users = set(zip(events["experiment_id"], events["user_id"], strict=True))
    if not event_users.issubset(assigned_users):
        raise ValueError("Найдены события пользователей без назначения в эксперимент.")


def _stage_rows(assignments: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    validate_inputs(assignments, events)
    base = assignments[
        [
            "experiment_id",
            "user_id",
            "variant",
            "assigned_epoch_s",
            "data_complete_through_day_utc",
        ]
    ].copy()
    keys = ["experiment_id", "user_id"]
    event_subset = events[
        ["experiment_id", "user_id", "event_name", "event_epoch_s"]
    ].merge(base[keys + ["assigned_epoch_s"]], on=keys, how="inner")
    event_subset = event_subset.loc[
        event_subset["event_epoch_s"].between(
            event_subset["assigned_epoch_s"],
            event_subset["assigned_epoch_s"] + WINDOW_SECONDS,
            inclusive="both",
        )
        & event_subset["event_name"].isin(FUNNEL_STAGES)
    ].copy()

    rows = base.set_index(keys)
    landing = (
        event_subset.loc[event_subset["event_name"].eq("landing_view")]
        .groupby(keys)["event_epoch_s"]
        .min()
        .rename("valid_landing_view")
    )
    rows = rows.join(landing)

    previous_stage = "valid_landing_view"
    for stage in ("signup", "activation", "purchase"):
        candidates = event_subset.loc[
            event_subset["event_name"].eq(stage), keys + ["event_epoch_s"]
        ].merge(
            rows[[previous_stage]].reset_index(), on=keys, how="inner"
        )
        candidates = candidates.loc[
            candidates[previous_stage].notna()
            & candidates["event_epoch_s"].gt(candidates[previous_stage])
        ]
        next_stage = f"valid_{stage}"
        stage_minima = (
            candidates.groupby(keys)["event_epoch_s"].min().rename(next_stage)
        )
        rows = rows.join(stage_minima)
        previous_stage = next_stage
    return rows.reset_index()


def analyze_funnel(assignments: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = _stage_rows(assignments, events)
    records: list[dict[str, Any]] = []
    for (experiment_id, variant), group in rows.groupby(
        ["experiment_id", "variant"], sort=True
    ):
        record: dict[str, Any] = {
            "experiment_id": experiment_id,
            "variant": variant,
            "n_assigned": int(group["user_id"].nunique()),
        }
        for stage in FUNNEL_STAGES:
            record[f"n_{stage}"] = int(group[f"valid_{stage}"].notna().sum())
        records.append(record)
    return pd.DataFrame(records).sort_values(
        ["experiment_id", "variant"], kind="stable"
    ).reset_index(drop=True)


def analyze_retention(assignments: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = _stage_rows(assignments, events)
    signups = rows.loc[
        rows["valid_signup"].notna(),
        [
            "experiment_id",
            "user_id",
            "variant",
            "valid_signup",
            "data_complete_through_day_utc",
        ],
    ].copy()
    signups["cohort_day_utc"] = (signups["valid_signup"] // 86_400).astype("int64")
    signups = signups.loc[
        signups["data_complete_through_day_utc"] >= signups["cohort_day_utc"] + 7
    ].copy()

    active = events.loc[
        events["event_name"].eq("active"),
        ["experiment_id", "user_id", "event_day_utc"],
    ].drop_duplicates()
    active = active.merge(
        signups[["experiment_id", "user_id", "cohort_day_utc"]],
        on=["experiment_id", "user_id"],
        how="inner",
    )
    active["is_d1"] = active["event_day_utc"].eq(active["cohort_day_utc"] + 1)
    active["is_d7"] = active["event_day_utc"].eq(active["cohort_day_utc"] + 7)
    flags = (
        active.groupby(["experiment_id", "user_id"], as_index=False)[["is_d1", "is_d7"]]
        .max()
    )
    for column in ("is_d1", "is_d7"):
        flags[column] = flags[column].astype("boolean")
    signups = signups.merge(
        flags, on=["experiment_id", "user_id"], how="left"
    )
    for column in ("is_d1", "is_d7"):
        signups[column] = signups[column].astype("boolean").fillna(False).astype(bool)

    result = (
        signups.groupby(["experiment_id", "cohort_day_utc", "variant"], as_index=False)
        .agg(
            cohort_users=("user_id", "nunique"),
            d1_users=("is_d1", "sum"),
            d7_users=("is_d7", "sum"),
        )
        .sort_values(["experiment_id", "cohort_day_utc", "variant"], kind="stable")
        .reset_index(drop=True)
    )
    for column in ("cohort_users", "d1_users", "d7_users"):
        result[column] = result[column].astype("int64")
    return result


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Ожидаются 0 <= successes <= total и total > 0.")
    z = float(norm.ppf(1 - (1 - confidence) / 2))
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = (
        z
        * np.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return float(center - half), float(center + half)


def newcombe_difference_interval(
    control_successes: int,
    control_total: int,
    treatment_successes: int,
    treatment_total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Интервал Ньюкомба для разности независимых долей (treatment − control)."""
    p_control = control_successes / control_total
    p_treatment = treatment_successes / treatment_total
    control_low, control_high = wilson_interval(
        control_successes, control_total, confidence
    )
    treatment_low, treatment_high = wilson_interval(
        treatment_successes, treatment_total, confidence
    )
    difference = p_treatment - p_control
    lower = difference - np.sqrt(
        (p_treatment - treatment_low) ** 2 + (control_high - p_control) ** 2
    )
    upper = difference + np.sqrt(
        (treatment_high - p_treatment) ** 2 + (p_control - control_low) ** 2
    )
    return float(lower), float(upper)


def analyze_experiment(
    assignments: pd.DataFrame,
    funnel: pd.DataFrame,
    alpha: float = ALPHA,
    srm_alpha: float = SRM_ALPHA,
) -> dict[str, Any]:
    if set(assignments["variant"].unique()) != {"control", "treatment"}:
        raise ValueError("Для A/B-анализа требуются обе группы: control и treatment.")
    assignment_counts = assignments.groupby("variant")["user_id"].nunique()
    n_control = int(assignment_counts["control"])
    n_treatment = int(assignment_counts["treatment"])
    expected = (n_control + n_treatment) / 2
    srm_statistic = (n_control - expected) ** 2 / expected + (
        n_treatment - expected
    ) ** 2 / expected
    srm_p_value = float(chi2.sf(srm_statistic, df=1))

    purchase_counts = funnel.set_index("variant")["n_purchase"]
    control_successes = int(purchase_counts["control"])
    treatment_successes = int(purchase_counts["treatment"])
    p_control = control_successes / n_control
    p_treatment = treatment_successes / n_treatment
    difference = p_treatment - p_control
    pooled = (control_successes + treatment_successes) / (n_control + n_treatment)
    pooled_se = np.sqrt(pooled * (1 - pooled) * (1 / n_control + 1 / n_treatment))
    z_statistic = float(difference / pooled_se) if pooled_se else 0.0
    p_value = float(2 * norm.sf(abs(z_statistic)))
    ci_low, ci_high = newcombe_difference_interval(
        control_successes, n_control, treatment_successes, n_treatment
    )

    return {
        "primary_metric": "Доля назначенных пользователей с покупкой в течение 7 дней",
        "primary_event": "purchase",
        "direction": "treatment − control",
        "alpha": alpha,
        "srm_alpha": srm_alpha,
        "srm_statistic": float(srm_statistic),
        "srm_p_value": srm_p_value,
        "srm_detected": srm_p_value < srm_alpha,
        "n_control": n_control,
        "n_treatment": n_treatment,
        "control_purchases": control_successes,
        "treatment_purchases": treatment_successes,
        "control_rate": p_control,
        "treatment_rate": p_treatment,
        "absolute_difference": difference,
        "relative_difference": difference / p_control if p_control else None,
        "ci_95_newcombe": [ci_low, ci_high],
        "ci_lower_at_least_mde": ci_low >= MDE_ABSOLUTE,
        "z_statistic": z_statistic,
        "p_value_two_sided": p_value,
        "statistically_significant": p_value < alpha,
        "mde_absolute": MDE_ABSOLUTE,
        "estimate_at_least_mde": difference >= MDE_ABSOLUTE,
    }


def analyze_all(
    assignments: pd.DataFrame, events: pd.DataFrame, seed: int = DEFAULT_SEED
) -> dict[str, Any]:
    validate_inputs(assignments, events)
    funnel = analyze_funnel(assignments, events)
    retention = analyze_retention(assignments, events)
    experiment = analyze_experiment(assignments, funnel)
    retention_summary = (
        retention.groupby(["experiment_id", "variant"], as_index=False)[
            ["cohort_users", "d1_users", "d7_users"]
        ]
        .sum()
        .sort_values(["experiment_id", "variant"], kind="stable")
    )
    power = {
        "baseline_rate": BASELINE_PURCHASE_RATE,
        "absolute_mde": MDE_ABSOLUTE,
        "alpha_two_sided": ALPHA,
        "target_power": TARGET_POWER,
        "required_n_per_arm_unrounded": REQUIRED_N_PER_ARM,
        "planned_n_per_arm": int(experiment["n_control"]),
        "planned_total_n": int(experiment["n_control"] + experiment["n_treatment"]),
        "achieved_approximate_power_at_planned_n": estimated_power(
            min(experiment["n_control"], experiment["n_treatment"])
        ),
        "method": "двусторонняя нормальная аппроксимация мощности для двух независимых долей",
    }
    return {
        "simulation": {
            "is_synthetic": True,
            "seed": seed,
            "experiment_id": str(assignments["experiment_id"].iloc[0]),
            "n_users": int(assignments["user_id"].nunique()),
            "n_events": int(len(events)),
            "data_complete_through_day_utc": int(
                assignments["data_complete_through_day_utc"].max()
            ),
        },
        "power_plan": power,
        "funnel": funnel.to_dict(orient="records"),
        "retention_by_cohort": retention.to_dict(orient="records"),
        "retention_summary": retention_summary.to_dict(orient="records"),
        "experiment": experiment,
    }


def verify_sql_parity(
    assignments: pd.DataFrame,
    events: pd.DataFrame,
    project_root: Path,
) -> dict[str, str]:
    """Исполняет PostgreSQL-совместимые запросы в SQLite и сравнивает с pandas.

    Запросы используют стандартные CTE, CASE, MIN, MAX и целочисленные UTC
    epoch/day поля, поэтому тот же SQL исполняется и в PostgreSQL, и в SQLite.
    """
    validate_inputs(assignments, events)
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        connection.executescript((project_root / "sql" / "01_schema.sql").read_text(encoding="utf-8"))

        assignment_columns = [
            "experiment_id",
            "user_id",
            "variant",
            "assigned_at",
            "assigned_epoch_s",
            "assigned_day_utc",
            "data_complete_through_day_utc",
        ]
        event_columns = [
            "event_id",
            "experiment_id",
            "user_id",
            "event_name",
            "event_ts",
            "event_epoch_s",
            "event_day_utc",
        ]
        connection.executemany(
            "INSERT INTO experiment_assignments "
            "(experiment_id, user_id, variant, assigned_at, assigned_epoch_s, "
            "assigned_day_utc, data_complete_through_day_utc) VALUES (?, ?, ?, ?, ?, ?, ?)",
            assignments[assignment_columns].itertuples(index=False, name=None),
        )
        connection.executemany(
            "INSERT INTO events "
            "(event_id, experiment_id, user_id, event_name, event_ts, event_epoch_s, "
            "event_day_utc) VALUES (?, ?, ?, ?, ?, ?, ?)",
            events[event_columns].itertuples(index=False, name=None),
        )

        funnel_sql = (project_root / "sql" / "02_funnel.sql").read_text(encoding="utf-8")
        retention_sql = (project_root / "sql" / "03_retention.sql").read_text(encoding="utf-8")
        srm_sql = (project_root / "sql" / "04_srm.sql").read_text(encoding="utf-8")
        funnel_sql_rows = pd.read_sql_query(funnel_sql, connection)
        retention_sql_rows = pd.read_sql_query(retention_sql, connection)
        srm_sql_rows = pd.read_sql_query(srm_sql, connection)

        funnel_python = analyze_funnel(assignments, events)
        retention_python = analyze_retention(assignments, events)
        funnel_columns = ["experiment_id", "variant", "n_assigned"] + [
            f"n_{stage}" for stage in FUNNEL_STAGES
        ]
        retention_columns = [
            "experiment_id",
            "cohort_day_utc",
            "variant",
            "cohort_users",
            "d1_users",
            "d7_users",
        ]
        funnel_sql_rows = funnel_sql_rows[funnel_columns].sort_values(
            ["experiment_id", "variant"], kind="stable"
        ).reset_index(drop=True)
        funnel_python = funnel_python[funnel_columns].sort_values(
            ["experiment_id", "variant"], kind="stable"
        ).reset_index(drop=True)
        retention_sql_rows = retention_sql_rows[retention_columns].sort_values(
            ["experiment_id", "cohort_day_utc", "variant"], kind="stable"
        ).reset_index(drop=True)
        retention_python = retention_python[retention_columns].sort_values(
            ["experiment_id", "cohort_day_utc", "variant"], kind="stable"
        ).reset_index(drop=True)

        pd.testing.assert_frame_equal(
            funnel_sql_rows, funnel_python, check_dtype=False, check_like=False
        )
        pd.testing.assert_frame_equal(
            retention_sql_rows, retention_python, check_dtype=False, check_like=False
        )
        expected_srm = (
            assignments.groupby(["experiment_id"], as_index=False)
            .agg(
                n_assigned=("user_id", "nunique"),
                n_control=("variant", lambda values: int((values == "control").sum())),
                n_treatment=(
                    "variant",
                    lambda values: int((values == "treatment").sum()),
                ),
            )
            .sort_values("experiment_id", kind="stable")
            .reset_index(drop=True)
        )
        srm_sql_rows = srm_sql_rows[
            ["experiment_id", "n_assigned", "n_control", "n_treatment"]
        ].sort_values("experiment_id", kind="stable").reset_index(drop=True)
        pd.testing.assert_frame_equal(srm_sql_rows, expected_srm, check_dtype=False)
    finally:
        connection.close()
    return {
        "funnel": "совпадает: SQL (SQLite, PostgreSQL-совместимый текст) = pandas",
        "retention": "совпадает: SQL (SQLite, PostgreSQL-совместимый текст) = pandas",
        "srm_assignment_counts": "совпадает: SQL = pandas",
    }


def write_results(
    results: dict[str, Any],
    results_json: Path,
    results_markdown: Path,
) -> None:
    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_json.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    experiment = results["experiment"]
    ci_low, ci_high = experiment["ci_95_newcombe"]
    funnel_lines = [
        "| Вариант | Назначено | Визит | Регистрация | Активация | Покупка |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in results["funnel"]:
        funnel_lines.append(
            "| {variant} | {n_assigned:,} | {n_landing_view:,} | {n_signup:,} | "
            "{n_activation:,} | {n_purchase:,} |".format(**row).replace(",", " ")
        )
    retention_lines = [
        "| Вариант | Когорта регистрации | Активны D1 | Активны D7 |",
        "|---|---:|---:|---:|",
    ]
    for row in results["retention_summary"]:
        d1_rate = row["d1_users"] / row["cohort_users"] if row["cohort_users"] else 0
        d7_rate = row["d7_users"] / row["cohort_users"] if row["cohort_users"] else 0
        retention_lines.append(
            f"| {row['variant']} | {row['cohort_users']:,} | "
            f"{row['d1_users']:,} ({d1_rate:.1%}) | "
            f"{row['d7_users']:,} ({d7_rate:.1%}) |".replace(",", " ")
        )
    report = f"""# Результаты воспроизводимого прогона

**Это симуляция, а не данные продукта.** Seed: {results['simulation']['seed']}. В выборке {results['simulation']['n_users']:,} назначенных пользователей и {results['simulation']['n_events']:,} событий. Генератор задаёт 12,0% конверсии control и 14,4% treatment в среднем; фактические числа ниже случайны.

## План до эксперимента

Базовая конверсия — 12,0%; MDE — +2,4 процентного пункта; двусторонний α = 0,05; мощность = 80%. Нормальная аппроксимация двух независимых долей дала минимум {results['power_plan']['required_n_per_arm_unrounded']:,} пользователей на вариант. Для проекта зафиксировано {results['power_plan']['planned_n_per_arm']:,} на вариант; расчётная мощность при таком размере — {results['power_plan']['achieved_approximate_power_at_planned_n']:.1%}.

## Последовательная воронка за 7 дней

Знаменатель на каждом этапе — все один раз назначенные пользователи варианта. Этап засчитывается, только если первое допустимое событие произошло строго после предыдущего и не позднее 7 суток от назначения.

{chr(10).join(funnel_lines)}

## Когортное удержание

Когорта — UTC-дата первой валидной регистрации. D1/D7 — наличие хотя бы одного события active на календарный день регистрации +1/+7 UTC. В знаменатель входят зарегистрировавшиеся пользователи только из полностью наблюдаемых когорт.

{chr(10).join(retention_lines)}

Полная таблица по каждой дате когорты находится в reports/results.json.

## Заранее заданный A/B-тест

Основная метрика — покупка в течение 7 суток после назначения на пользователя из знаменателя всех назначенных. Назначение 1:1, одно назначение на пару experiment_id/user_id. SRM проверен χ²-тестом при пороге p < 0,001. Для долей приведён двусторонний pooled z-тест и 95% интервал Ньюкомба для разности treatment − control.

- Control: {experiment['control_purchases']:,}/{experiment['n_control']:,} = {experiment['control_rate']:.2%}.
- Treatment: {experiment['treatment_purchases']:,}/{experiment['n_treatment']:,} = {experiment['treatment_rate']:.2%}.
- Разность: {experiment['absolute_difference']:+.2%} ({experiment['absolute_difference'] * 100:+.2f} п.п.); 95% ДИ Ньюкомба [{ci_low:+.2%}; {ci_high:+.2%}].
- z = {experiment['z_statistic']:.3f}; двустороннее p = {experiment['p_value_two_sided']:.4g}; статистически значимо при α=0,05: **{'да' if experiment['statistically_significant'] else 'нет'}**.
- SRM: χ² = {experiment['srm_statistic']:.3f}; p = {experiment['srm_p_value']:.4g}; сигнал SRM при пороге 0,001: **{'да' if experiment['srm_detected'] else 'нет'}**.
- Плановый MDE равен +{MDE_ABSOLUTE * 100:.1f} п.п. Точечная оценка {'достигает' if experiment['estimate_at_least_mde'] else 'не достигает'} этого порога; нижняя граница ДИ {'выше' if experiment['ci_lower_at_least_mde'] else 'ниже'} MDE. MDE — плановый порог, а не универсальная оценка ценности.

Результат описывает только случайный прогон генератора. Он показывает работу процедуры расчёта, но не оценивает и не доказывает причинный эффект реального продукта. Статистическая значимость сама по себе не означает практическую ценность.

## Проверки

{chr(10).join(f'- {name}: {status}.' for name, status in results['sql_parity'].items())}
- Последовательность этапов, уникальность пользователя в знаменателе, границы окна и зрелость D7 проверены unit-тестами.
"""
    results_markdown.write_text(report, encoding="utf-8")

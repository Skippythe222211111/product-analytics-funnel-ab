"""Генератор воспроизводимых синтетических событий."""

from __future__ import annotations

from datetime import datetime, timezone
from math import ceil, sqrt

import numpy as np
import pandas as pd
from scipy.stats import norm


EXPERIMENT_ID = "onboarding_purchase_2026_01"
DEFAULT_SEED = 20261103
START_EPOCH_S = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())
RECRUITMENT_DAYS = 7
MAX_SIGNUP_DELAY_DAYS = 2
RETENTION_DAYS = 7
BASELINE_PURCHASE_RATE = 0.12
MDE_ABSOLUTE = 0.024
ALPHA = 0.05
TARGET_POWER = 0.80
SRM_ALPHA = 0.001
ACTIVATION_GIVEN_SIGNUP = 0.50
SIGNUP_RATE = 0.70
D1_RETENTION_RATE = 0.32
D7_RETENTION_RATE = 0.18


def estimated_power(
    n_per_arm: int,
    p_control: float = BASELINE_PURCHASE_RATE,
    p_treatment: float = BASELINE_PURCHASE_RATE + MDE_ABSOLUTE,
    alpha: float = ALPHA,
) -> float:
    """Двусторонняя нормальная аппроксимация мощности для двух долей."""
    if n_per_arm < 1:
        return 0.0
    z_critical = norm.ppf(1 - alpha / 2)
    standard_error = sqrt(
        p_control * (1 - p_control) / n_per_arm
        + p_treatment * (1 - p_treatment) / n_per_arm
    )
    noncentrality = abs(p_treatment - p_control) / standard_error
    return float(
        norm.cdf(-z_critical - noncentrality)
        + norm.sf(z_critical - noncentrality)
    )


def required_sample_size_per_arm(
    p_control: float = BASELINE_PURCHASE_RATE,
    absolute_mde: float = MDE_ABSOLUTE,
    alpha: float = ALPHA,
    target_power: float = TARGET_POWER,
) -> int:
    """Минимальный размер одной группы методом целочисленного поиска."""
    p_treatment = p_control + absolute_mde
    if not (0 < p_control < p_treatment < 1):
        raise ValueError("Нужны доли 0 < p_control < p_control + MDE < 1.")
    if not (0 < alpha < 1 and 0 < target_power < 1):
        raise ValueError("alpha и target_power должны находиться между 0 и 1.")

    low, high = 1, 1
    while estimated_power(high, p_control, p_treatment, alpha) < target_power:
        high *= 2
    while low < high:
        middle = (low + high) // 2
        if estimated_power(middle, p_control, p_treatment, alpha) >= target_power:
            high = middle
        else:
            low = middle + 1
    return low


REQUIRED_N_PER_ARM = required_sample_size_per_arm()
N_PER_ARM = ceil(REQUIRED_N_PER_ARM / 100) * 100
DEFAULT_N_USERS = 2 * N_PER_ARM


def _iso_utc(epoch_seconds: int) -> str:
    return datetime.fromtimestamp(int(epoch_seconds), tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _append_events(
    rows: list[dict[str, object]],
    user_ids: np.ndarray,
    event_name: str,
    epochs: np.ndarray,
    experiment_id: str,
    discriminator: str = "",
) -> None:
    for user_id, event_epoch in zip(user_ids, epochs, strict=True):
        suffix = f":{discriminator}" if discriminator else ""
        event_id = f"{user_id}:{event_name}{suffix}"
        epoch = int(event_epoch)
        rows.append(
            {
                "event_id": event_id,
                "experiment_id": experiment_id,
                "user_id": str(user_id),
                "event_name": event_name,
                "event_ts": _iso_utc(epoch),
                "event_epoch_s": epoch,
                "event_day_utc": epoch // 86_400,
            }
        )


def generate_synthetic_data(
    n_users: int = DEFAULT_N_USERS,
    seed: int = DEFAULT_SEED,
    experiment_id: str = EXPERIMENT_ID,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Создаёт сбалансированные назначения и события без внешних данных.

    Вариант назначается случайной перестановкой с фиксированным seed.
    В симуляции у treatment задана конверсия в покупку 14,4% против 12,0%
    у control; это параметр сценария, а не оценка эффекта реального продукта.
    """
    if n_users < 2 or n_users % 2:
        raise ValueError("Для сбалансированного теста n_users должно быть чётным и >= 2.")

    rng = np.random.default_rng(seed)
    user_ids = np.array([f"u{number:07d}" for number in range(1, n_users + 1)])

    permutation = rng.permutation(n_users)
    variants = np.empty(n_users, dtype=object)
    variants[permutation[: n_users // 2]] = "control"
    variants[permutation[n_users // 2 :]] = "treatment"

    recruitment_day = rng.integers(0, RECRUITMENT_DAYS, size=n_users)
    second_of_day = rng.integers(0, 86_400, size=n_users)
    assigned_epochs = START_EPOCH_S + recruitment_day * 86_400 + second_of_day
    start_day_utc = START_EPOCH_S // 86_400
    data_complete_through_day_utc = (
        start_day_utc + RECRUITMENT_DAYS - 1 + MAX_SIGNUP_DELAY_DAYS + RETENTION_DAYS
    )

    assignments = pd.DataFrame(
        {
            "experiment_id": experiment_id,
            "user_id": user_ids,
            "variant": variants,
            "assigned_at": [_iso_utc(epoch) for epoch in assigned_epochs],
            "assigned_epoch_s": assigned_epochs.astype(np.int64),
            "assigned_day_utc": (assigned_epochs // 86_400).astype(np.int64),
            "data_complete_through_day_utc": data_complete_through_day_utc,
        }
    )

    event_rows: list[dict[str, object]] = []
    _append_events(event_rows, user_ids, "landing_view", assigned_epochs, experiment_id)

    signup_mask = rng.random(n_users) < SIGNUP_RATE
    signup_indices = np.flatnonzero(signup_mask)
    signup_epochs = assigned_epochs[signup_indices] + rng.integers(
        1, MAX_SIGNUP_DELAY_DAYS * 86_400 + 1, size=len(signup_indices)
    )
    _append_events(
        event_rows,
        user_ids[signup_indices],
        "signup",
        signup_epochs,
        experiment_id,
    )

    activation_mask = rng.random(len(signup_indices)) < ACTIVATION_GIVEN_SIGNUP
    activation_indices = signup_indices[activation_mask]
    activation_epochs = signup_epochs[activation_mask] + rng.integers(
        3_600, 48 * 3_600 + 1, size=len(activation_indices)
    )
    _append_events(
        event_rows,
        user_ids[activation_indices],
        "activation",
        activation_epochs,
        experiment_id,
    )

    purchase_probability = np.where(
        variants[activation_indices] == "treatment",
        (BASELINE_PURCHASE_RATE + MDE_ABSOLUTE)
        / (SIGNUP_RATE * ACTIVATION_GIVEN_SIGNUP),
        BASELINE_PURCHASE_RATE / (SIGNUP_RATE * ACTIVATION_GIVEN_SIGNUP),
    )
    purchase_mask = rng.random(len(activation_indices)) < purchase_probability
    purchase_indices = activation_indices[purchase_mask]
    purchase_epochs = activation_epochs[purchase_mask] + rng.integers(
        3_600, 72 * 3_600 + 1, size=len(purchase_indices)
    )
    _append_events(
        event_rows,
        user_ids[purchase_indices],
        "purchase",
        purchase_epochs,
        experiment_id,
    )

    for retention_day, probability, label in (
        (1, D1_RETENTION_RATE, "D1"),
        (7, D7_RETENTION_RATE, "D7"),
    ):
        retained_mask = rng.random(len(signup_indices)) < probability
        retained_indices = signup_indices[retained_mask]
        signup_days = signup_epochs[retained_mask] // 86_400
        activity_epochs = signup_days * 86_400 + retention_day * 86_400 + 12 * 3_600
        _append_events(
            event_rows,
            user_ids[retained_indices],
            "active",
            activity_epochs,
            experiment_id,
            discriminator=label,
        )

    events = pd.DataFrame(event_rows)
    events = events.sort_values(
        ["event_epoch_s", "user_id", "event_name", "event_id"], kind="stable"
    ).reset_index(drop=True)
    assignments = assignments.sort_values("user_id", kind="stable").reset_index(drop=True)
    return assignments, events

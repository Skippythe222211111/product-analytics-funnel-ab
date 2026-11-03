-- Когорта = UTC-дата первой валидной регистрации; D1/D7 — календарный день.
-- Незрелые D7-когорты исключаются по известной полноте наблюдения.
WITH bounds AS (
    SELECT
        experiment_id, user_id, variant, assigned_epoch_s,
        assigned_epoch_s + 604800 AS window_end_epoch_s,
        data_complete_through_day_utc
    FROM experiment_assignments
),
landing AS (
    SELECT
        b.experiment_id, b.user_id, b.variant, b.assigned_epoch_s,
        b.window_end_epoch_s, b.data_complete_through_day_utc,
        MIN(CASE WHEN e.event_name = 'landing_view'
            AND e.event_epoch_s BETWEEN b.assigned_epoch_s AND b.window_end_epoch_s
            THEN e.event_epoch_s END) AS landing_epoch_s
    FROM bounds b
    LEFT JOIN events e
        ON e.experiment_id = b.experiment_id AND e.user_id = b.user_id
    GROUP BY b.experiment_id, b.user_id, b.variant, b.assigned_epoch_s,
             b.window_end_epoch_s, b.data_complete_through_day_utc
),
signups AS (
    SELECT
        l.experiment_id, l.user_id, l.variant, l.data_complete_through_day_utc,
        MIN(CASE WHEN e.event_name = 'signup'
            AND e.event_epoch_s > l.landing_epoch_s
            AND e.event_epoch_s <= l.window_end_epoch_s
            THEN e.event_epoch_s END) AS signup_epoch_s
    FROM landing l
    LEFT JOIN events e
        ON e.experiment_id = l.experiment_id AND e.user_id = l.user_id
    GROUP BY l.experiment_id, l.user_id, l.variant,
             l.data_complete_through_day_utc, l.landing_epoch_s,
             l.window_end_epoch_s
),
cohorts AS (
    SELECT
        experiment_id, user_id, variant,
        signup_epoch_s / 86400 AS cohort_day_utc
    FROM signups
    WHERE signup_epoch_s IS NOT NULL
      AND data_complete_through_day_utc >= signup_epoch_s / 86400 + 7
),
user_retention AS (
    SELECT
        c.experiment_id, c.user_id, c.variant, c.cohort_day_utc,
        MAX(CASE WHEN e.event_name = 'active'
            AND e.event_day_utc = c.cohort_day_utc + 1 THEN 1 ELSE 0 END) AS is_d1,
        MAX(CASE WHEN e.event_name = 'active'
            AND e.event_day_utc = c.cohort_day_utc + 7 THEN 1 ELSE 0 END) AS is_d7
    FROM cohorts c
    LEFT JOIN events e
        ON e.experiment_id = c.experiment_id AND e.user_id = c.user_id
    GROUP BY c.experiment_id, c.user_id, c.variant, c.cohort_day_utc
)
SELECT
    experiment_id,
    cohort_day_utc,
    variant,
    COUNT(*) AS cohort_users,
    SUM(is_d1) AS d1_users,
    SUM(is_d7) AS d7_users
FROM user_retention
GROUP BY experiment_id, cohort_day_utc, variant
ORDER BY experiment_id, cohort_day_utc, variant;

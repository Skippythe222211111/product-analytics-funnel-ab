-- Последовательная воронка на пользователя, окно [назначение, назначение + 7 суток].
-- Одинаковые события сворачиваются до самого раннего подходящего времени.
WITH bounds AS (
    SELECT
        experiment_id,
        user_id,
        variant,
        assigned_epoch_s,
        assigned_epoch_s + 604800 AS window_end_epoch_s
    FROM experiment_assignments
),
landing AS (
    SELECT
        b.experiment_id, b.user_id, b.variant, b.assigned_epoch_s,
        b.window_end_epoch_s,
        MIN(CASE WHEN e.event_name = 'landing_view'
            AND e.event_epoch_s BETWEEN b.assigned_epoch_s AND b.window_end_epoch_s
            THEN e.event_epoch_s END) AS landing_epoch_s
    FROM bounds b
    LEFT JOIN events e
        ON e.experiment_id = b.experiment_id AND e.user_id = b.user_id
    GROUP BY b.experiment_id, b.user_id, b.variant,
             b.assigned_epoch_s, b.window_end_epoch_s
),
signup AS (
    SELECT
        l.experiment_id, l.user_id, l.variant, l.assigned_epoch_s,
        l.window_end_epoch_s, l.landing_epoch_s,
        MIN(CASE WHEN e.event_name = 'signup'
            AND e.event_epoch_s > l.landing_epoch_s
            AND e.event_epoch_s <= l.window_end_epoch_s
            THEN e.event_epoch_s END) AS signup_epoch_s
    FROM landing l
    LEFT JOIN events e
        ON e.experiment_id = l.experiment_id AND e.user_id = l.user_id
    GROUP BY l.experiment_id, l.user_id, l.variant, l.assigned_epoch_s,
             l.window_end_epoch_s, l.landing_epoch_s
),
activation AS (
    SELECT
        s.experiment_id, s.user_id, s.variant, s.assigned_epoch_s,
        s.window_end_epoch_s, s.landing_epoch_s, s.signup_epoch_s,
        MIN(CASE WHEN e.event_name = 'activation'
            AND e.event_epoch_s > s.signup_epoch_s
            AND e.event_epoch_s <= s.window_end_epoch_s
            THEN e.event_epoch_s END) AS activation_epoch_s
    FROM signup s
    LEFT JOIN events e
        ON e.experiment_id = s.experiment_id AND e.user_id = s.user_id
    GROUP BY s.experiment_id, s.user_id, s.variant, s.assigned_epoch_s,
             s.window_end_epoch_s, s.landing_epoch_s, s.signup_epoch_s
),
purchase AS (
    SELECT
        a.experiment_id, a.user_id, a.variant, a.assigned_epoch_s,
        a.window_end_epoch_s, a.landing_epoch_s, a.signup_epoch_s,
        a.activation_epoch_s,
        MIN(CASE WHEN e.event_name = 'purchase'
            AND e.event_epoch_s > a.activation_epoch_s
            AND e.event_epoch_s <= a.window_end_epoch_s
            THEN e.event_epoch_s END) AS purchase_epoch_s
    FROM activation a
    LEFT JOIN events e
        ON e.experiment_id = a.experiment_id AND e.user_id = a.user_id
    GROUP BY a.experiment_id, a.user_id, a.variant, a.assigned_epoch_s,
             a.window_end_epoch_s, a.landing_epoch_s, a.signup_epoch_s,
             a.activation_epoch_s
)
SELECT
    experiment_id,
    variant,
    COUNT(*) AS n_assigned,
    SUM(CASE WHEN landing_epoch_s IS NOT NULL THEN 1 ELSE 0 END) AS n_landing_view,
    SUM(CASE WHEN signup_epoch_s IS NOT NULL THEN 1 ELSE 0 END) AS n_signup,
    SUM(CASE WHEN activation_epoch_s IS NOT NULL THEN 1 ELSE 0 END) AS n_activation,
    SUM(CASE WHEN purchase_epoch_s IS NOT NULL THEN 1 ELSE 0 END) AS n_purchase
FROM purchase
GROUP BY experiment_id, variant
ORDER BY experiment_id, variant;

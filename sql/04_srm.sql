-- Фактические размеры назначенных групп для последующей проверки SRM в Python.
SELECT
    experiment_id,
    COUNT(*) AS n_assigned,
    SUM(CASE WHEN variant = 'control' THEN 1 ELSE 0 END) AS n_control,
    SUM(CASE WHEN variant = 'treatment' THEN 1 ELSE 0 END) AS n_treatment
FROM experiment_assignments
GROUP BY experiment_id
ORDER BY experiment_id;

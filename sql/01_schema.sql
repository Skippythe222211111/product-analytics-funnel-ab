-- Схема PostgreSQL. Все временные метки хранятся в UTC.
-- Epoch/day колонки заполняются загрузчиком как UTC-производные от timestamp;
-- они обеспечивают одинаковую арифметику в PostgreSQL и SQLite.
CREATE TABLE IF NOT EXISTS experiment_assignments (
    experiment_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    variant TEXT NOT NULL CHECK (variant IN ('control', 'treatment')),
    assigned_at TIMESTAMPTZ NOT NULL,
    assigned_epoch_s BIGINT NOT NULL,
    assigned_day_utc BIGINT NOT NULL,
    data_complete_through_day_utc BIGINT NOT NULL,
    PRIMARY KEY (experiment_id, user_id),
    CHECK (data_complete_through_day_utc >= assigned_day_utc)
);

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    event_name TEXT NOT NULL CHECK (
        event_name IN ('landing_view', 'signup', 'activation', 'purchase', 'active')
    ),
    event_ts TIMESTAMPTZ NOT NULL,
    event_epoch_s BIGINT NOT NULL,
    event_day_utc BIGINT NOT NULL,
    FOREIGN KEY (experiment_id, user_id)
        REFERENCES experiment_assignments (experiment_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_events_experiment_user_time
    ON events (experiment_id, user_id, event_epoch_s);
CREATE INDEX IF NOT EXISTS idx_events_experiment_name_day
    ON events (experiment_id, event_name, event_day_utc);

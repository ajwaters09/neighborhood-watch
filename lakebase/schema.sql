-- Lakebase (Postgres) schema for Neighborhood Watch: the app's operational tables.
--
-- Run once against the `public` schema of the project's database (Lakebase SQL Editor:
-- Branch overview -> SQL Editor). The trend tables the app also reads are Synced Tables from
-- Unity Catalog (notebook 04), not created here.
--
-- Every table gets REPLICA IDENTITY FULL: Lakebase Change Data Feed needs whole old and new row
-- images in the write-ahead log to land each table's history in Unity Catalog as
-- lb_<table>_history (notebooks 06 and 07 read those).

CREATE TABLE users (
  user_id       BIGSERIAL PRIMARY KEY,
  email         TEXT UNIQUE NOT NULL,
  display_name  TEXT,
  created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE area_subscriptions (
  subscription_id BIGSERIAL PRIMARY KEY,
  user_id         BIGINT REFERENCES users(user_id),
  community_area  INT NOT NULL,
  created_at      TIMESTAMPTZ DEFAULT now(),
  UNIQUE (user_id, community_area)
);

CREATE TABLE resident_reports (
  report_id       BIGSERIAL PRIMARY KEY,
  user_id         BIGINT REFERENCES users(user_id),
  community_area  INT NOT NULL,
  description     TEXT NOT NULL,
  created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE alert_rules (
  rule_id         BIGSERIAL PRIMARY KEY,
  user_id         BIGINT REFERENCES users(user_id),
  community_area  INT NOT NULL,
  metric          TEXT NOT NULL,      -- e.g. 'crime_total', '311_streetlight_out'
  threshold_pct   NUMERIC NOT NULL,   -- trailing-window % change that triggers the rule
  created_at      TIMESTAMPTZ DEFAULT now(),
  -- 'threshold' flags when the metric is up threshold_pct or more; 'trend' ("Alert me on this
  -- trend") flags on every new data pull, whatever the numbers do (webapp/data_access.py).
  kind            TEXT NOT NULL DEFAULT 'threshold' CHECK (kind IN ('threshold', 'trend')),
  -- The trend data's as-of date when the rule was set or its alert last cleared: the rule only
  -- fires on data newer than this. NULL fires on any data.
  quiet_through   DATE,
  -- The metric's % change at that same moment, which a trend update compares against.
  last_pct        NUMERIC
);

CREATE INDEX idx_area_subscriptions_community_area ON area_subscriptions(community_area);
CREATE INDEX idx_alert_rules_community_area ON alert_rules(community_area);

ALTER TABLE users REPLICA IDENTITY FULL;
ALTER TABLE area_subscriptions REPLICA IDENTITY FULL;
ALTER TABLE resident_reports REPLICA IDENTITY FULL;
ALTER TABLE alert_rules REPLICA IDENTITY FULL;

-- Every agent tool call, written by @_logged in src/agent_tools.py: UI-driven and chat-driven
-- calls, reads and failures included (subscription_events only sees successful writes).
CREATE TABLE tool_invocations (
  invocation_id   BIGSERIAL PRIMARY KEY,
  tool_name       TEXT NOT NULL,
  user_id         BIGINT REFERENCES users(user_id),  -- NULL: read tools take no user_id
  community_area  INT,
  success         BOOLEAN NOT NULL,
  error_message   TEXT,
  created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_tool_invocations_tool_name ON tool_invocations(tool_name);
CREATE INDEX idx_tool_invocations_created_at ON tool_invocations(created_at);
ALTER TABLE tool_invocations REPLICA IDENTITY FULL;

-- One row per chat turn, for analytics and evals (notebook 07): the question, every tool call
-- (name, arguments, ok/error, latency), the final answer, total latency, and an optional
-- thumbs up/down. Written best-effort by the app's /chat route (data_access.log_chat_turn).
CREATE TABLE chat_turns (
  turn_id           BIGSERIAL PRIMARY KEY,
  chat_id           TEXT NOT NULL,          -- one browser session's conversation
  user_id           BIGINT REFERENCES users(user_id),
  turn_index        INT NOT NULL,           -- 1-based position within chat_id
  active_area       INT,                    -- area open in the app when asked
  user_message      TEXT NOT NULL,
  assistant_message TEXT,
  tool_calls        JSONB,                  -- [{name, arguments, ok, error, latency_ms}]
  raw_messages      JSONB,                  -- full assistant/tool messages, for replay and evals
  model             TEXT,
  latency_ms        INT,
  error             TEXT,
  feedback          SMALLINT CHECK (feedback IN (-1, 1)),
  feedback_at       TIMESTAMPTZ,
  created_at        TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_chat_turns_created_at ON chat_turns(created_at);
CREATE INDEX idx_chat_turns_chat_id ON chat_turns(chat_id);
ALTER TABLE chat_turns REPLICA IDENTITY FULL;

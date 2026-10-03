-- Migration for a database created before trend alerts and unsubscribing (schema.sql already
-- includes this for a fresh one). Run it in the Lakebase SQL Editor as the project owner BEFORE
-- deploying the app code that reads these columns. Replace <APP_SP> as in grant_app_access.sql.
--
-- alert_rules gains:
-- - kind: 'threshold' (flags when the metric is up threshold_pct or more, as every rule did) or
--   'trend' (flags on every new data pull). Existing rules become 'threshold'.
-- - quiet_through: the trend data's as-of date when a rule was set or its alert last cleared; a
--   rule fires only on newer data. Existing rules keep NULL and fire on the current data.
-- - last_pct: the metric's % change at that moment, which a trend update compares against.
-- The app gains UPDATE on alert_rules (clearing an alert) and DELETE on area_subscriptions.

ALTER TABLE alert_rules ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'threshold'
  CHECK (kind IN ('threshold', 'trend'));
ALTER TABLE alert_rules ADD COLUMN IF NOT EXISTS quiet_through DATE;
ALTER TABLE alert_rules ADD COLUMN IF NOT EXISTS last_pct NUMERIC;

GRANT UPDATE ON public.alert_rules TO "<APP_SP>";
GRANT DELETE ON public.area_subscriptions TO "<APP_SP>";

-- Check: the new columns are there.
SELECT column_name, data_type FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = 'alert_rules' ORDER BY ordinal_position;

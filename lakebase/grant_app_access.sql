-- Grants the deployed Databricks App's service principal what it needs in Lakebase.
--
-- A notebook connects as its user, who owns the project. The deployed app runs as its own service
-- principal, which starts with no privileges at all (its Unity Catalog grants are in
-- docs/operations.md).
--
-- Before running: replace every <APP_SP> with the app's service principal application ID (App
-- details -> Authorization). Run in the Lakebase SQL Editor as the project owner.
--
-- The role must come from databricks_create_role(): a plain CREATE ROLE (or the REST API) makes a
-- password role, which the OAuth-only endpoint rejects. This one is bound to the principal's
-- Databricks identity, so the token src/db_connect.py mints is accepted.

SELECT databricks_create_role('<APP_SP>');

-- Reads: the trend Synced Tables (in the schema src/config.LAKEBASE_SYNC_SCHEMA names) and the
-- app's own tables.
GRANT SELECT ON ajwaters_chicago.area_trend_metrics_lb TO "<APP_SP>";
GRANT SELECT ON ajwaters_chicago.area_trend_rolling_lb TO "<APP_SP>";
GRANT SELECT ON public.users TO "<APP_SP>";
GRANT SELECT ON public.area_subscriptions TO "<APP_SP>";
GRANT SELECT ON public.resident_reports TO "<APP_SP>";
GRANT SELECT ON public.alert_rules TO "<APP_SP>";

-- Writes: sign-up, subscribe and unsubscribe, reports, alert rules (create, clear, delete).
-- INSERT ... RETURNING and an UPDATE's WHERE also need SELECT, granted above.
GRANT INSERT, UPDATE ON public.users TO "<APP_SP>";
GRANT INSERT, DELETE ON public.area_subscriptions TO "<APP_SP>";
GRANT INSERT ON public.resident_reports TO "<APP_SP>";
GRANT INSERT, UPDATE, DELETE ON public.alert_rules TO "<APP_SP>";

-- Logging. A missing grant here fails silently (logging is best-effort), except the chat
-- feedback buttons, which show an error toast.
GRANT INSERT ON public.tool_invocations TO "<APP_SP>";
GRANT SELECT, INSERT, UPDATE ON public.chat_turns TO "<APP_SP>";

-- The BIGSERIAL keys' sequences.
GRANT USAGE ON SEQUENCE public.users_user_id_seq TO "<APP_SP>";
GRANT USAGE ON SEQUENCE public.area_subscriptions_subscription_id_seq TO "<APP_SP>";
GRANT USAGE ON SEQUENCE public.resident_reports_report_id_seq TO "<APP_SP>";
GRANT USAGE ON SEQUENCE public.alert_rules_rule_id_seq TO "<APP_SP>";
GRANT USAGE ON SEQUENCE public.tool_invocations_invocation_id_seq TO "<APP_SP>";
GRANT USAGE ON SEQUENCE public.chat_turns_turn_id_seq TO "<APP_SP>";

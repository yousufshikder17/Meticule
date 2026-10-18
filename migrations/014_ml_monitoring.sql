BEGIN;
SET LOCAL search_path=public;

-- Reference distribution of one model version: bin edges or top categories from its training partition, with counts.
-- Built once (on first use) and never changed, so every drift evaluation of the version compares against the same bins.
CREATE TABLE ml_drift_references (
  tenant_id uuid NOT NULL, model_version_id uuid NOT NULL, profile jsonb NOT NULL, rows integer NOT NULL CHECK(rows>0),
  created_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(tenant_id,model_version_id),
  FOREIGN KEY(tenant_id,model_version_id) REFERENCES ml_model_versions(tenant_id,id)
);
-- Per completed prediction: counts per reference bin only. Raw inputs are never stored. Rows may be pruned by the owner
-- role (ml_prune_monitoring); the API role can neither change nor delete them.
CREATE TABLE ml_prediction_profiles (
  tenant_id uuid NOT NULL, prediction_id uuid NOT NULL, endpoint_id uuid NOT NULL, model_version_id uuid NOT NULL,
  rows integer NOT NULL CHECK(rows>0), profile jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY(tenant_id,prediction_id), FOREIGN KEY(tenant_id,prediction_id) REFERENCES ml_predictions(tenant_id,id),
  FOREIGN KEY(tenant_id,model_version_id) REFERENCES ml_model_versions(tenant_id,id)
);
CREATE INDEX ml_prediction_profiles_window_idx ON ml_prediction_profiles(tenant_id,model_version_id,created_at);
-- Ground truth for one completed prediction, aligned with its rows. One submission per prediction.
CREATE TABLE ml_prediction_feedback (
  tenant_id uuid NOT NULL, prediction_id uuid NOT NULL, labels jsonb NOT NULL CHECK(jsonb_typeof(labels)='array'),
  created_by uuid NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY(tenant_id,prediction_id), FOREIGN KEY(tenant_id,prediction_id) REFERENCES ml_predictions(tenant_id,id)
);
-- A persisted drift evaluation: its window, thresholds, sample counts and per-feature evidence. Re-running the same window
-- over unchanged profiles reproduces it.
CREATE TABLE ml_drift_evaluations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(), tenant_id uuid NOT NULL, deployment_id uuid NOT NULL, model_version_id uuid NOT NULL,
  window_start timestamptz NOT NULL, window_end timestamptz NOT NULL, thresholds jsonb NOT NULL, observed_rows bigint NOT NULL CHECK(observed_rows>=0),
  reference_rows integer NOT NULL, status text NOT NULL CHECK(status IN ('stable','moderate','significant','insufficient_data')),
  features jsonb NOT NULL, output jsonb, created_by uuid NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(tenant_id,id), CHECK(window_start<window_end),
  FOREIGN KEY(tenant_id,deployment_id) REFERENCES ml_deployments(tenant_id,id), FOREIGN KEY(tenant_id,model_version_id) REFERENCES ml_model_versions(tenant_id,id)
);
CREATE INDEX ml_drift_evaluations_idx ON ml_drift_evaluations(tenant_id,deployment_id,created_at DESC);

DO $$ DECLARE t text; BEGIN
  FOREACH t IN ARRAY ARRAY['ml_drift_references','ml_prediction_profiles','ml_prediction_feedback','ml_drift_evaluations'] LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY',t);
    EXECUTE format('CREATE POLICY tenant_isolation ON %I TO durable_agent_api USING (tenant_id = nullif(current_setting(''app.tenant_id'',true),'''')::uuid) WITH CHECK (tenant_id = nullif(current_setting(''app.tenant_id'',true),'''')::uuid)',t);
    EXECUTE format('GRANT SELECT,INSERT ON %I TO durable_agent_api',t);
  END LOOP;
  FOREACH t IN ARRAY ARRAY['ml_drift_references','ml_prediction_feedback','ml_drift_evaluations'] LOOP
    EXECUTE format('CREATE TRIGGER ml_append_only BEFORE UPDATE OR DELETE ON %I FOR EACH ROW EXECUTE FUNCTION reject_evaluation_evidence_update()',t);
  END LOOP;
END $$;
CREATE FUNCTION ml_profile_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'Prediction profiles are immutable' USING ERRCODE='check_violation'; END $$;
CREATE TRIGGER ml_profile_no_update BEFORE UPDATE ON ml_prediction_profiles FOR EACH ROW EXECUTE FUNCTION ml_profile_immutable();

-- Retention for operators (run as the owner role, e.g. from a scheduled job): removes profiles older than `keep`.
-- Prediction records, feedback and evaluations are kept.
CREATE FUNCTION ml_prune_monitoring(keep interval) RETURNS bigint LANGUAGE sql AS $$
  WITH removed AS (DELETE FROM ml_prediction_profiles WHERE created_at < now() - keep RETURNING 1) SELECT count(*) FROM removed;
$$;
REVOKE EXECUTE ON FUNCTION ml_prune_monitoring(interval) FROM PUBLIC;

INSERT INTO schema_migrations(version) VALUES('014_ml_monitoring');
COMMIT;

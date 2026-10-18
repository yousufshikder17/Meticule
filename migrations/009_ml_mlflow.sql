BEGIN;
SET LOCAL search_path=public;

-- One export record per attempt and tracking server (the server is identified by a hash of its configured URI, never the
-- URI itself). The unique key serialises concurrent exports of the same attempt; MLflow-side tags make retries idempotent.
CREATE TABLE ml_mlflow_exports (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(), tenant_id uuid NOT NULL, training_run_id uuid NOT NULL,
  tracking_uri_hash text NOT NULL CHECK(tracking_uri_hash ~ '^[a-f0-9]{64}$'),
  status text NOT NULL CHECK(status IN ('PENDING','COMPLETED','FAILED')),
  mlflow_experiment_id text, mlflow_run_id text, unsupported jsonb NOT NULL DEFAULT '[]', failure jsonb,
  attempts integer NOT NULL DEFAULT 0 CHECK(attempts>=0), created_by uuid NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(tenant_id,id), UNIQUE(tenant_id,training_run_id,tracking_uri_hash),
  FOREIGN KEY(tenant_id,training_run_id) REFERENCES ml_training_runs(tenant_id,id),
  CHECK((status='COMPLETED')=(mlflow_run_id IS NOT NULL))
);
-- A completed export is final; only pending/failed records are retried.
CREATE FUNCTION ml_mlflow_export_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF (NEW.id,NEW.tenant_id,NEW.training_run_id,NEW.tracking_uri_hash,NEW.created_by,NEW.created_at) IS DISTINCT FROM (OLD.id,OLD.tenant_id,OLD.training_run_id,OLD.tracking_uri_hash,OLD.created_by,OLD.created_at)
     OR OLD.status='COMPLETED' THEN
    RAISE EXCEPTION 'MLflow export identity and completed exports are immutable' USING ERRCODE='check_violation';
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER ml_mlflow_export_update BEFORE UPDATE ON ml_mlflow_exports FOR EACH ROW EXECUTE FUNCTION ml_mlflow_export_guard();

-- Imported MLflow run metadata: parameters, metrics, tags, status and times as data. No artifact or model is ever imported.
CREATE TABLE ml_external_runs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(), tenant_id uuid NOT NULL, experiment_id uuid NOT NULL,
  source text NOT NULL CHECK(source='mlflow'), tracking_uri_hash text NOT NULL CHECK(tracking_uri_hash ~ '^[a-f0-9]{64}$'),
  external_run_id text NOT NULL CHECK(external_run_id ~ '^[0-9a-f]{32}$'), document jsonb NOT NULL,
  imported_by uuid NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(tenant_id,id), UNIQUE(tenant_id,experiment_id,tracking_uri_hash,external_run_id),
  FOREIGN KEY(tenant_id,experiment_id) REFERENCES ml_experiments(tenant_id,id)
);
CREATE TRIGGER ml_external_run_append_only BEFORE UPDATE OR DELETE ON ml_external_runs FOR EACH ROW EXECUTE FUNCTION reject_evaluation_evidence_update();

DO $$ DECLARE t text; BEGIN
  FOREACH t IN ARRAY ARRAY['ml_mlflow_exports','ml_external_runs'] LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY',t);
    EXECUTE format('CREATE POLICY tenant_isolation ON %I TO durable_agent_api USING (tenant_id = nullif(current_setting(''app.tenant_id'',true),'''')::uuid) WITH CHECK (tenant_id = nullif(current_setting(''app.tenant_id'',true),'''')::uuid)',t);
    EXECUTE format('GRANT SELECT,INSERT ON %I TO durable_agent_api',t);
  END LOOP;
END $$;
GRANT UPDATE ON ml_mlflow_exports TO durable_agent_api;

INSERT INTO schema_migrations(version) VALUES('009_ml_mlflow');
COMMIT;

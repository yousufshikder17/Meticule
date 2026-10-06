BEGIN;
SET LOCAL search_path = public;
DO $$ BEGIN
  IF EXISTS(SELECT 1 FROM schema_migrations WHERE version='002_ml') THEN
    RAISE EXCEPTION 'Unreleased ML precursor detected; export its data before migrating to ML V1';
  END IF;
END $$;

ALTER TABLE runs ADD COLUMN kind text NOT NULL DEFAULT 'agent' CHECK (kind IN ('agent','ml'));
ALTER TABLE runs ALTER COLUMN agent_id DROP NOT NULL;
ALTER TABLE runs ALTER COLUMN agent_version DROP NOT NULL;
ALTER TABLE runs ALTER COLUMN agent_configuration_snapshot DROP NOT NULL;
ALTER TABLE runs ADD CONSTRAINT runs_workload_shape CHECK (
  (kind='agent' AND agent_id IS NOT NULL AND agent_version IS NOT NULL AND agent_configuration_snapshot IS NOT NULL) OR
  (kind='ml' AND agent_id IS NULL AND agent_version IS NULL AND agent_configuration_snapshot IS NULL AND parent_run_id IS NULL)
);
CREATE INDEX runs_workload_queue_idx ON runs(kind,created_at,id) WHERE status='queued';
-- Existing recovery requeues expired running work; ordinary callers still cannot do so.
CREATE OR REPLACE FUNCTION enforce_run_transition() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF NEW.status=OLD.status THEN RETURN NEW; END IF;
  IF OLD.kind='ml' AND OLD.status='running' AND NEW.status='queued' AND OLD.lease_expires_at<=now() AND NEW.lease_owner IS NULL THEN RETURN NEW; END IF;
  IF NOT (CASE OLD.status
    WHEN 'queued' THEN NEW.status IN ('claimed','cancelling')
    WHEN 'claimed' THEN NEW.status IN ('running','queued','cancelling')
    WHEN 'running' THEN NEW.status IN ('waiting_for_approval','paused','cancelling','completed','failed')
    WHEN 'waiting_for_approval' THEN NEW.status IN ('queued','cancelling')
    WHEN 'paused' THEN NEW.status IN ('queued','cancelling')
    WHEN 'cancelling' THEN NEW.status IN ('cancelled','failed') ELSE false END) THEN
    RAISE EXCEPTION 'invalid run transition: % -> %',OLD.status,NEW.status USING ERRCODE='check_violation';
  END IF;
  RETURN NEW;
END $$;
ALTER TABLE usage_records ADD COLUMN usage_kind text NOT NULL DEFAULT 'model' CHECK (usage_kind IN ('model','ml_training'));
ALTER TABLE usage_records ADD COLUMN duration_ms bigint CHECK (duration_ms >= 0);

-- Public bounded storage: no shared-filesystem or object-store requirement.
CREATE TABLE ml_blobs (
  tenant_id uuid NOT NULL, content_hash text NOT NULL CHECK(content_hash ~ '^[a-f0-9]{64}$'),
  media_type text NOT NULL, content bytea NOT NULL CHECK(octet_length(content) BETWEEN 1 AND 16000000),
  created_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(tenant_id,content_hash)
);
CREATE TRIGGER ml_blob_append_only BEFORE UPDATE OR DELETE ON ml_blobs FOR EACH ROW EXECUTE FUNCTION reject_evaluation_evidence_update();

CREATE TABLE ml_datasets (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(), tenant_id uuid NOT NULL, name text NOT NULL,
  created_by uuid NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), UNIQUE(tenant_id,id)
);
CREATE TABLE ml_dataset_versions (
  id uuid PRIMARY KEY, tenant_id uuid NOT NULL, dataset_id uuid NOT NULL, version integer NOT NULL CHECK(version>0),
  document jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), UNIQUE(tenant_id,id), UNIQUE(tenant_id,dataset_id,version),
  FOREIGN KEY(tenant_id,dataset_id) REFERENCES ml_datasets(tenant_id,id)
);
CREATE TABLE ml_feature_pipelines (
  id uuid PRIMARY KEY, tenant_id uuid NOT NULL, logical_id uuid NOT NULL, version integer NOT NULL CHECK(version>0),
  document jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), UNIQUE(tenant_id,id), UNIQUE(tenant_id,logical_id,version)
);
CREATE TABLE ml_experiments (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(), tenant_id uuid NOT NULL, name text NOT NULL,
  created_by uuid NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), UNIQUE(tenant_id,id)
);
CREATE TABLE ml_training_jobs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(), tenant_id uuid NOT NULL, run_id uuid NOT NULL UNIQUE, experiment_id uuid NOT NULL,
  dataset_version_id uuid NOT NULL, pipeline_id uuid NOT NULL, snapshot jsonb NOT NULL,
  status text NOT NULL DEFAULT 'QUEUED' CHECK(status IN ('QUEUED','PREPARING','TRAINING','EVALUATING','SAVING','COMPLETED','FAILED','CANCELLED')),
  created_at timestamptz NOT NULL DEFAULT now(), UNIQUE(tenant_id,id),
  FOREIGN KEY(tenant_id,run_id) REFERENCES runs(tenant_id,id),
  FOREIGN KEY(tenant_id,experiment_id) REFERENCES ml_experiments(tenant_id,id),
  FOREIGN KEY(tenant_id,dataset_version_id) REFERENCES ml_dataset_versions(tenant_id,id),
  FOREIGN KEY(tenant_id,pipeline_id) REFERENCES ml_feature_pipelines(tenant_id,id)
);
CREATE TABLE ml_training_runs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(), tenant_id uuid NOT NULL, job_id uuid NOT NULL, attempt integer NOT NULL CHECK(attempt>0),
  snapshot jsonb NOT NULL, status text NOT NULL DEFAULT 'QUEUED' CHECK(status IN ('QUEUED','PREPARING','TRAINING','EVALUATING','SAVING','COMPLETED','FAILED','CANCELLED')),
  environment jsonb NOT NULL DEFAULT '{}', split_indices jsonb NOT NULL DEFAULT '{}', resolved_hyperparameters jsonb NOT NULL DEFAULT '{}',
  started_at timestamptz NOT NULL DEFAULT now(), ended_at timestamptz, failure jsonb,
  UNIQUE(tenant_id,id), UNIQUE(tenant_id,job_id,attempt), FOREIGN KEY(tenant_id,job_id) REFERENCES ml_training_jobs(tenant_id,id),
  CHECK ((status IN ('COMPLETED','FAILED','CANCELLED')) = (ended_at IS NOT NULL))
);
CREATE TABLE ml_metrics (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(), tenant_id uuid NOT NULL, training_run_id uuid NOT NULL,
  name text NOT NULL, partition text NOT NULL CHECK(partition IN ('train','validation','test')),
  value double precision NOT NULL CHECK(value > '-Infinity'::float8 AND value < 'Infinity'::float8),
  direction text NOT NULL CHECK(direction IN ('higher','lower')), created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(tenant_id,training_run_id,name,partition), FOREIGN KEY(tenant_id,training_run_id) REFERENCES ml_training_runs(tenant_id,id)
);
CREATE TABLE ml_artifacts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(), tenant_id uuid NOT NULL, training_run_id uuid NOT NULL,
  kind text NOT NULL CHECK(kind IN ('model','checkpoint','manifest')), format text NOT NULL, content jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(), UNIQUE(tenant_id,id), UNIQUE(tenant_id,training_run_id,kind),
  FOREIGN KEY(tenant_id,training_run_id) REFERENCES ml_training_runs(tenant_id,id)
);
CREATE TABLE ml_registry_entries (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(), tenant_id uuid NOT NULL, name text NOT NULL, created_by uuid NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(), UNIQUE(tenant_id,id), UNIQUE(tenant_id,name)
);
CREATE TABLE ml_model_versions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(), tenant_id uuid NOT NULL, registry_entry_id uuid NOT NULL,
  training_run_id uuid NOT NULL, artifact_id uuid NOT NULL, version integer NOT NULL CHECK(version>0),
  status text NOT NULL DEFAULT 'REGISTERED' CHECK(status IN ('REGISTERED','READY','RETIRED','REJECTED')),
  created_at timestamptz NOT NULL DEFAULT now(), UNIQUE(tenant_id,id), UNIQUE(tenant_id,registry_entry_id,version),
  FOREIGN KEY(tenant_id,registry_entry_id) REFERENCES ml_registry_entries(tenant_id,id),
  FOREIGN KEY(tenant_id,training_run_id) REFERENCES ml_training_runs(tenant_id,id),
  FOREIGN KEY(tenant_id,artifact_id) REFERENCES ml_artifacts(tenant_id,id)
);
CREATE TABLE ml_endpoints (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(), tenant_id uuid NOT NULL, name text NOT NULL, model_version_id uuid NOT NULL,
  status text NOT NULL DEFAULT 'ACTIVE' CHECK(status IN ('ACTIVE','DISABLED')), created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(tenant_id,id), FOREIGN KEY(tenant_id,model_version_id) REFERENCES ml_model_versions(tenant_id,id)
);
CREATE TABLE ml_predictions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(), tenant_id uuid NOT NULL, endpoint_id uuid NOT NULL, model_version_id uuid NOT NULL,
  status text NOT NULL DEFAULT 'PENDING' CHECK(status IN ('PENDING','COMPLETED','FAILED')), input_hash text NOT NULL,
  count integer NOT NULL CHECK(count>0), latency_ms bigint CHECK(latency_ms>=0), output jsonb, failure jsonb,
  created_at timestamptz NOT NULL DEFAULT now(), UNIQUE(tenant_id,id),
  FOREIGN KEY(tenant_id,endpoint_id) REFERENCES ml_endpoints(tenant_id,id),
  FOREIGN KEY(tenant_id,model_version_id) REFERENCES ml_model_versions(tenant_id,id)
);
CREATE INDEX ml_predictions_monitor_idx ON ml_predictions(tenant_id,endpoint_id,created_at);

CREATE FUNCTION ml_training_transition_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF NEW.snapshot IS DISTINCT FROM OLD.snapshot OR NEW.tenant_id<>OLD.tenant_id OR NEW.id<>OLD.id THEN
    RAISE EXCEPTION 'ML training provenance is immutable' USING ERRCODE='check_violation';
  END IF;
  IF TG_TABLE_NAME='ml_training_jobs' THEN
    IF (NEW.run_id,NEW.experiment_id,NEW.dataset_version_id,NEW.pipeline_id) IS DISTINCT FROM (OLD.run_id,OLD.experiment_id,OLD.dataset_version_id,OLD.pipeline_id) THEN RAISE EXCEPTION 'ML job identity is immutable'; END IF;
    -- Only canonical lease recovery may reset job progress; attempt evidence never resets.
    IF NEW.status='QUEUED' AND OLD.status NOT IN ('QUEUED','COMPLETED','FAILED','CANCELLED') AND EXISTS(SELECT 1 FROM runs WHERE id=NEW.run_id AND status='queued') THEN RETURN NEW; END IF;
  ELSE
    IF (NEW.job_id,NEW.attempt,NEW.started_at) IS DISTINCT FROM (OLD.job_id,OLD.attempt,OLD.started_at) THEN RAISE EXCEPTION 'ML attempt identity is immutable'; END IF;
    IF OLD.status IN ('COMPLETED','FAILED','CANCELLED') AND NEW IS DISTINCT FROM OLD THEN RAISE EXCEPTION 'Terminal ML evidence is immutable'; END IF;
  END IF;
  IF NEW.status=OLD.status THEN RETURN NEW; END IF;
  IF NOT (CASE OLD.status
    WHEN 'QUEUED' THEN NEW.status IN ('PREPARING','FAILED','CANCELLED')
    WHEN 'PREPARING' THEN NEW.status IN ('TRAINING','FAILED','CANCELLED')
    WHEN 'TRAINING' THEN NEW.status IN ('EVALUATING','FAILED','CANCELLED')
    WHEN 'EVALUATING' THEN NEW.status IN ('SAVING','FAILED','CANCELLED')
    WHEN 'SAVING' THEN NEW.status IN ('COMPLETED','FAILED','CANCELLED') ELSE false END) THEN
    RAISE EXCEPTION 'invalid ML training transition: % -> %',OLD.status,NEW.status USING ERRCODE='check_violation';
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER ml_jobs_transition BEFORE UPDATE ON ml_training_jobs FOR EACH ROW EXECUTE FUNCTION ml_training_transition_guard();
CREATE TRIGGER ml_runs_transition BEFORE UPDATE ON ml_training_runs FOR EACH ROW EXECUTE FUNCTION ml_training_transition_guard();

CREATE FUNCTION ml_sync_canonical_run() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE next_status text;
BEGIN
  IF NEW.kind<>'ml' OR NEW.status=OLD.status THEN RETURN NEW; END IF;
  IF NEW.status='completed' AND NOT EXISTS(SELECT 1 FROM ml_training_jobs j JOIN ml_training_runs t ON t.tenant_id=j.tenant_id AND t.job_id=j.id WHERE j.run_id=NEW.id AND j.status='SAVING' AND t.status='SAVING') THEN RAISE EXCEPTION 'ML completion requires a saving attempt'; END IF;
  next_status := CASE NEW.status WHEN 'completed' THEN 'COMPLETED' WHEN 'failed' THEN 'FAILED' WHEN 'cancelled' THEN 'CANCELLED' ELSE NULL END;
  IF next_status IS NOT NULL THEN
    UPDATE ml_training_runs t SET status=next_status,ended_at=now(),failure=CASE WHEN next_status='COMPLETED' THEN NULL ELSE COALESCE(NEW.error_details,jsonb_build_object('code',lower(next_status))) END
      FROM ml_training_jobs j WHERE t.tenant_id=j.tenant_id AND t.job_id=j.id AND j.run_id=NEW.id AND t.ended_at IS NULL;
    UPDATE ml_training_jobs SET status=next_status WHERE run_id=NEW.id;
  ELSIF NEW.status='queued' THEN
    UPDATE ml_training_runs t SET status='FAILED',ended_at=now(),failure='{"code":"lease_lost"}'
      FROM ml_training_jobs j WHERE t.tenant_id=j.tenant_id AND t.job_id=j.id AND j.run_id=NEW.id AND t.ended_at IS NULL;
    UPDATE ml_training_jobs SET status='QUEUED' WHERE run_id=NEW.id;
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER ml_canonical_sync AFTER UPDATE OF status ON runs FOR EACH ROW EXECUTE FUNCTION ml_sync_canonical_run();

CREATE FUNCTION ml_model_transition_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF (NEW.id,NEW.tenant_id,NEW.registry_entry_id,NEW.training_run_id,NEW.artifact_id,NEW.version) IS DISTINCT FROM (OLD.id,OLD.tenant_id,OLD.registry_entry_id,OLD.training_run_id,OLD.artifact_id,OLD.version) THEN RAISE EXCEPTION 'Model provenance is immutable'; END IF;
  IF NEW.status=OLD.status THEN RETURN NEW; END IF;
  IF NOT ((OLD.status='REGISTERED' AND NEW.status IN ('READY','REJECTED')) OR (OLD.status='READY' AND NEW.status='RETIRED')) THEN RAISE EXCEPTION 'invalid ML model transition' USING ERRCODE='check_violation'; END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER ml_model_transition BEFORE UPDATE ON ml_model_versions FOR EACH ROW EXECUTE FUNCTION ml_model_transition_guard();

DO $$ DECLARE t text; BEGIN
  FOREACH t IN ARRAY ARRAY['ml_blobs','ml_datasets','ml_dataset_versions','ml_feature_pipelines','ml_experiments','ml_training_jobs','ml_training_runs','ml_metrics','ml_artifacts','ml_registry_entries','ml_model_versions','ml_endpoints','ml_predictions'] LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY',t);
    EXECUTE format('CREATE POLICY tenant_isolation ON %I TO durable_agent_api USING (tenant_id = nullif(current_setting(''app.tenant_id'',true),'''')::uuid) WITH CHECK (tenant_id = nullif(current_setting(''app.tenant_id'',true),'''')::uuid)',t);
    EXECUTE format('GRANT SELECT,INSERT ON %I TO durable_agent_api',t);
  END LOOP;
  FOREACH t IN ARRAY ARRAY['ml_dataset_versions','ml_feature_pipelines','ml_metrics','ml_artifacts'] LOOP
    EXECUTE format('CREATE TRIGGER ml_append_only BEFORE UPDATE OR DELETE ON %I FOR EACH ROW EXECUTE FUNCTION reject_evaluation_evidence_update()',t);
  END LOOP;
END $$;
-- Canonical cancellation/recovery triggers execute with the caller's privileges.
GRANT UPDATE ON ml_training_jobs,ml_training_runs,ml_model_versions,ml_endpoints,ml_predictions TO durable_agent_api;
INSERT INTO schema_migrations(version) VALUES('002_ml_foundation');
COMMIT;

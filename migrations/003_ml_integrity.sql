BEGIN;
SET LOCAL search_path=public;

-- Schema IDs denote immutable definitions, even across dataset versions.
CREATE TABLE ml_dataset_schemas (
  id uuid NOT NULL, tenant_id uuid NOT NULL, document jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY(tenant_id,id)
);
INSERT INTO ml_dataset_schemas(id,tenant_id,document)
  SELECT DISTINCT ON (tenant_id,document->'schema'->>'id') (document->'schema'->>'id')::uuid,tenant_id,document->'schema'
  FROM ml_dataset_versions ORDER BY tenant_id,document->'schema'->>'id',created_at;
ALTER TABLE ml_dataset_versions ADD COLUMN schema_id uuid;
ALTER TABLE ml_dataset_versions DISABLE TRIGGER ml_append_only;
UPDATE ml_dataset_versions SET schema_id=(document->'schema'->>'id')::uuid;
ALTER TABLE ml_dataset_versions ENABLE TRIGGER ml_append_only;
ALTER TABLE ml_dataset_versions ALTER COLUMN schema_id SET NOT NULL;
ALTER TABLE ml_dataset_versions ADD FOREIGN KEY(tenant_id,schema_id) REFERENCES ml_dataset_schemas(tenant_id,id);
ALTER TABLE ml_dataset_schemas ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON ml_dataset_schemas TO durable_agent_api
  USING(tenant_id=nullif(current_setting('app.tenant_id',true),'')::uuid)
  WITH CHECK(tenant_id=nullif(current_setting('app.tenant_id',true),'')::uuid);
GRANT SELECT,INSERT ON ml_dataset_schemas TO durable_agent_api;
CREATE TRIGGER ml_schema_append_only BEFORE UPDATE OR DELETE ON ml_dataset_schemas FOR EACH ROW EXECUTE FUNCTION reject_evaluation_evidence_update();

CREATE FUNCTION ml_evidence_insert_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF NOT EXISTS(SELECT 1 FROM ml_training_runs t WHERE t.tenant_id=NEW.tenant_id AND t.id=NEW.training_run_id AND t.status='SAVING') THEN
    RAISE EXCEPTION 'ML evidence requires a saving attempt' USING ERRCODE='check_violation';
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER ml_metric_insert BEFORE INSERT ON ml_metrics FOR EACH ROW EXECUTE FUNCTION ml_evidence_insert_guard();
CREATE TRIGGER ml_artifact_insert BEFORE INSERT ON ml_artifacts FOR EACH ROW EXECUTE FUNCTION ml_evidence_insert_guard();

CREATE FUNCTION ml_registry_insert_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF NEW.status<>'REGISTERED' OR NOT EXISTS(
    SELECT 1 FROM ml_training_runs t JOIN ml_artifacts a ON a.tenant_id=t.tenant_id AND a.training_run_id=t.id
    WHERE t.tenant_id=NEW.tenant_id AND t.id=NEW.training_run_id AND t.status='COMPLETED' AND a.id=NEW.artifact_id AND a.kind='model'
  ) THEN RAISE EXCEPTION 'Model registration requires its completed training artifact' USING ERRCODE='check_violation'; END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER ml_registry_insert BEFORE INSERT ON ml_model_versions FOR EACH ROW EXECUTE FUNCTION ml_registry_insert_guard();

CREATE FUNCTION ml_endpoint_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF TG_OP='UPDATE' THEN
    IF (NEW.id,NEW.tenant_id,NEW.model_version_id,NEW.created_at) IS DISTINCT FROM (OLD.id,OLD.tenant_id,OLD.model_version_id,OLD.created_at) OR (OLD.status='DISABLED' AND NEW.status<>'DISABLED') THEN RAISE EXCEPTION 'Endpoint model pin is immutable'; END IF;
  END IF;
  IF NEW.status='ACTIVE' AND NOT EXISTS(SELECT 1 FROM ml_model_versions m WHERE m.tenant_id=NEW.tenant_id AND m.id=NEW.model_version_id AND m.status='READY') THEN RAISE EXCEPTION 'Endpoint requires a ready model'; END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER ml_endpoint_integrity BEFORE INSERT OR UPDATE ON ml_endpoints FOR EACH ROW EXECUTE FUNCTION ml_endpoint_guard();

CREATE FUNCTION ml_prediction_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF TG_OP='INSERT' THEN
    IF NEW.status<>'PENDING' OR NOT EXISTS(SELECT 1 FROM ml_endpoints e WHERE e.tenant_id=NEW.tenant_id AND e.id=NEW.endpoint_id AND e.model_version_id=NEW.model_version_id AND e.status='ACTIVE') THEN RAISE EXCEPTION 'Prediction requires its active pinned endpoint'; END IF;
  ELSE
    IF (NEW.id,NEW.tenant_id,NEW.endpoint_id,NEW.model_version_id,NEW.input_hash,NEW.count,NEW.created_at) IS DISTINCT FROM (OLD.id,OLD.tenant_id,OLD.endpoint_id,OLD.model_version_id,OLD.input_hash,OLD.count,OLD.created_at) THEN RAISE EXCEPTION 'Prediction provenance is immutable'; END IF;
    IF OLD.status<>'PENDING' AND NEW IS DISTINCT FROM OLD THEN RAISE EXCEPTION 'Terminal prediction is immutable'; END IF;
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER ml_prediction_integrity BEFORE INSERT OR UPDATE ON ml_predictions FOR EACH ROW EXECUTE FUNCTION ml_prediction_guard();
ALTER TABLE ml_predictions ADD CHECK(input_hash ~ '^[a-f0-9]{64}$');
ALTER TABLE ml_predictions ADD CHECK(
  (status='PENDING' AND latency_ms IS NULL AND output IS NULL AND failure IS NULL) OR
  (status='COMPLETED' AND latency_ms IS NOT NULL AND output IS NOT NULL AND failure IS NULL) OR
  (status='FAILED' AND latency_ms IS NOT NULL AND output IS NULL AND failure IS NOT NULL)
);

CREATE FUNCTION ml_run_identity_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF NEW.kind<>OLD.kind THEN RAISE EXCEPTION 'Run workload kind is immutable'; END IF;
  IF NEW.kind='ml' AND (NEW.id,NEW.tenant_id,NEW.created_by,NEW.root_run_id) IS DISTINCT FROM (OLD.id,OLD.tenant_id,OLD.created_by,OLD.root_run_id) THEN RAISE EXCEPTION 'ML runtime identity is immutable'; END IF;
  IF NEW.kind='ml' AND NEW.status='completed' AND OLD.status<>'completed' AND NOT EXISTS(
    SELECT 1 FROM ml_training_jobs j JOIN ml_training_runs t ON t.tenant_id=j.tenant_id AND t.job_id=j.id
    JOIN ml_artifacts a ON a.tenant_id=t.tenant_id AND a.training_run_id=t.id AND a.kind='model'
    WHERE j.run_id=NEW.id AND t.status='SAVING' AND EXISTS(SELECT 1 FROM ml_metrics m WHERE m.tenant_id=t.tenant_id AND m.training_run_id=t.id)
  ) THEN RAISE EXCEPTION 'ML completion requires persisted model and metrics'; END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER ml_run_integrity BEFORE UPDATE ON runs FOR EACH ROW EXECUTE FUNCTION ml_run_identity_guard();

INSERT INTO schema_migrations(version) VALUES('003_ml_integrity');
COMMIT;

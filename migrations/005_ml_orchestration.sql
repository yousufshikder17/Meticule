BEGIN;
SET LOCAL search_path=public;

-- Benchmark, search and AutoML experiments group ordinary ML training jobs. Execution stays with the
-- canonical runs/leases; these tables only record which jobs belong together and the persisted outcome.
CREATE TABLE ml_orchestrations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(), tenant_id uuid NOT NULL, experiment_id uuid NOT NULL, parent_id uuid,
  kind text NOT NULL CHECK(kind IN ('benchmark','search','automl')),
  name text NOT NULL, config jsonb NOT NULL, config_hash text NOT NULL CHECK(config_hash ~ '^[a-f0-9]{64}$'),
  ranking_config jsonb NOT NULL, idempotency_key text CHECK(idempotency_key IS NULL OR length(idempotency_key) BETWEEN 1 AND 200),
  status text NOT NULL DEFAULT 'RUNNING' CHECK(status IN ('RUNNING','COMPLETED','PARTIALLY_COMPLETED','FAILED','CANCELLED')),
  ranking jsonb, cancel_requested_at timestamptz, finalized_at timestamptz,
  created_by uuid NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(tenant_id,id),
  FOREIGN KEY(tenant_id,experiment_id) REFERENCES ml_experiments(tenant_id,id),
  FOREIGN KEY(tenant_id,parent_id) REFERENCES ml_orchestrations(tenant_id,id),
  CHECK(kind<>'automl' OR parent_id IS NULL),
  CHECK((status='RUNNING')=(finalized_at IS NULL)),
  CHECK(status='RUNNING' OR ranking IS NOT NULL)
);
CREATE UNIQUE INDEX ml_orchestrations_idempotency_idx ON ml_orchestrations(tenant_id,idempotency_key) WHERE idempotency_key IS NOT NULL;
CREATE INDEX ml_orchestrations_list_idx ON ml_orchestrations(tenant_id,kind,created_at DESC);
CREATE INDEX ml_orchestrations_parent_idx ON ml_orchestrations(tenant_id,parent_id) WHERE parent_id IS NOT NULL;

CREATE TABLE ml_orchestration_members (
  tenant_id uuid NOT NULL, orchestration_id uuid NOT NULL, ordinal integer NOT NULL CHECK(ordinal>=0),
  job_id uuid NOT NULL, label text NOT NULL, candidate jsonb NOT NULL,
  PRIMARY KEY(tenant_id,orchestration_id,ordinal), UNIQUE(tenant_id,job_id),
  FOREIGN KEY(tenant_id,orchestration_id) REFERENCES ml_orchestrations(tenant_id,id),
  FOREIGN KEY(tenant_id,job_id) REFERENCES ml_training_jobs(tenant_id,id)
);

CREATE FUNCTION ml_orchestration_insert_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF NEW.status<>'RUNNING' THEN RAISE EXCEPTION 'Orchestrations start RUNNING' USING ERRCODE='check_violation'; END IF;
  IF NEW.parent_id IS NOT NULL AND NOT EXISTS(SELECT 1 FROM ml_orchestrations p WHERE p.tenant_id=NEW.tenant_id AND p.id=NEW.parent_id AND p.kind='automl' AND p.experiment_id=NEW.experiment_id AND p.status='RUNNING') THEN
    RAISE EXCEPTION 'Only a running AutoML experiment in the same experiment can parent an orchestration' USING ERRCODE='check_violation';
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER ml_orchestration_insert BEFORE INSERT ON ml_orchestrations FOR EACH ROW EXECUTE FUNCTION ml_orchestration_insert_guard();

CREATE FUNCTION ml_orchestration_update_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF OLD.status<>'RUNNING' THEN RAISE EXCEPTION 'Terminal orchestration is immutable' USING ERRCODE='check_violation'; END IF;
  IF (NEW.id,NEW.tenant_id,NEW.experiment_id,NEW.parent_id,NEW.kind,NEW.name,NEW.config,NEW.config_hash,NEW.ranking_config,NEW.idempotency_key,NEW.created_by,NEW.created_at)
     IS DISTINCT FROM (OLD.id,OLD.tenant_id,OLD.experiment_id,OLD.parent_id,OLD.kind,OLD.name,OLD.config,OLD.config_hash,OLD.ranking_config,OLD.idempotency_key,OLD.created_by,OLD.created_at) THEN
    RAISE EXCEPTION 'Orchestration definition is immutable' USING ERRCODE='check_violation';
  END IF;
  IF OLD.cancel_requested_at IS NOT NULL AND NEW.cancel_requested_at IS DISTINCT FROM OLD.cancel_requested_at THEN RAISE EXCEPTION 'Cancellation request is immutable' USING ERRCODE='check_violation'; END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER ml_orchestration_update BEFORE UPDATE ON ml_orchestrations FOR EACH ROW EXECUTE FUNCTION ml_orchestration_update_guard();

CREATE FUNCTION ml_member_insert_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF NOT EXISTS(SELECT 1 FROM ml_orchestrations o JOIN ml_training_jobs j ON j.tenant_id=o.tenant_id AND j.experiment_id=o.experiment_id
      WHERE o.tenant_id=NEW.tenant_id AND o.id=NEW.orchestration_id AND o.kind IN ('benchmark','search') AND o.status='RUNNING' AND j.id=NEW.job_id) THEN
    RAISE EXCEPTION 'Candidates join a running benchmark or search with a job from the same experiment' USING ERRCODE='check_violation';
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER ml_member_insert BEFORE INSERT ON ml_orchestration_members FOR EACH ROW EXECUTE FUNCTION ml_member_insert_guard();
CREATE TRIGGER ml_member_append_only BEFORE UPDATE OR DELETE ON ml_orchestration_members FOR EACH ROW EXECUTE FUNCTION reject_evaluation_evidence_update();

DO $$ DECLARE t text; BEGIN
  FOREACH t IN ARRAY ARRAY['ml_orchestrations','ml_orchestration_members'] LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY',t);
    EXECUTE format('CREATE POLICY tenant_isolation ON %I TO durable_agent_api USING (tenant_id = nullif(current_setting(''app.tenant_id'',true),'''')::uuid) WITH CHECK (tenant_id = nullif(current_setting(''app.tenant_id'',true),'''')::uuid)',t);
    EXECUTE format('GRANT SELECT,INSERT ON %I TO durable_agent_api',t);
  END LOOP;
END $$;
GRANT UPDATE ON ml_orchestrations TO durable_agent_api;

INSERT INTO schema_migrations(version) VALUES('005_ml_orchestration');
COMMIT;

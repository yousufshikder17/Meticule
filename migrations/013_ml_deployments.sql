BEGIN;
SET LOCAL search_path=public;

-- Endpoints gain an explicit runtime. An ONNX endpoint serves one VERIFIED export of its own pinned model version.
ALTER TABLE ml_endpoints ADD COLUMN runtime text NOT NULL DEFAULT 'native' CHECK(runtime IN ('native','onnx'));
ALTER TABLE ml_endpoints ADD COLUMN onnx_export_id uuid;
ALTER TABLE ml_endpoints ADD FOREIGN KEY(tenant_id,onnx_export_id) REFERENCES ml_onnx_exports(tenant_id,id);
ALTER TABLE ml_endpoints ADD CHECK((runtime='onnx')=(onnx_export_id IS NOT NULL));
CREATE FUNCTION ml_endpoint_runtime_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF TG_OP='UPDATE' AND (NEW.runtime,NEW.onnx_export_id) IS DISTINCT FROM (OLD.runtime,OLD.onnx_export_id) THEN RAISE EXCEPTION 'Endpoint model pin is immutable'; END IF;
  IF TG_OP='INSERT' AND NEW.runtime='onnx' AND NOT EXISTS(SELECT 1 FROM ml_onnx_exports x WHERE x.tenant_id=NEW.tenant_id AND x.id=NEW.onnx_export_id AND x.model_version_id=NEW.model_version_id AND x.status='VERIFIED') THEN
    RAISE EXCEPTION 'ONNX endpoints require a verified export of their model version' USING ERRCODE='check_violation';
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER ml_endpoint_runtime BEFORE INSERT OR UPDATE ON ml_endpoints FOR EACH ROW EXECUTE FUNCTION ml_endpoint_runtime_guard();

-- A deployment is a durable record of what should serve; each activation creates one pinned endpoint (the serving unit).
-- CREATED -> VALIDATING -> READY | FAILED; READY -> ACTIVE; ACTIVE -> INACTIVE; INACTIVE -> ACTIVE; READY, INACTIVE and
-- FAILED -> VALIDATING (a new revision). An ACTIVE deployment validates new revisions without leaving ACTIVE, and switching
-- its active revision (update or rollback) keeps it ACTIVE.
CREATE TABLE ml_deployments (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(), tenant_id uuid NOT NULL, name text NOT NULL CHECK(length(name) BETWEEN 1 AND 120),
  status text NOT NULL DEFAULT 'CREATED' CHECK(status IN ('CREATED','VALIDATING','READY','ACTIVE','INACTIVE','FAILED')),
  active_revision integer, active_endpoint_id uuid, created_by uuid NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(tenant_id,id), UNIQUE(tenant_id,name), FOREIGN KEY(tenant_id,active_endpoint_id) REFERENCES ml_endpoints(tenant_id,id),
  CHECK((status='ACTIVE')=(active_endpoint_id IS NOT NULL)), CHECK((active_revision IS NULL)=(active_endpoint_id IS NULL))
);
CREATE TABLE ml_deployment_revisions (
  tenant_id uuid NOT NULL, deployment_id uuid NOT NULL, revision integer NOT NULL CHECK(revision>0), model_version_id uuid NOT NULL,
  runtime text NOT NULL CHECK(runtime IN ('native','onnx')), onnx_export_id uuid,
  status text NOT NULL DEFAULT 'VALIDATING' CHECK(status IN ('VALIDATING','READY','FAILED')), validation jsonb,
  created_by uuid NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), validated_at timestamptz,
  PRIMARY KEY(tenant_id,deployment_id,revision), FOREIGN KEY(tenant_id,deployment_id) REFERENCES ml_deployments(tenant_id,id),
  FOREIGN KEY(tenant_id,model_version_id) REFERENCES ml_model_versions(tenant_id,id), FOREIGN KEY(tenant_id,onnx_export_id) REFERENCES ml_onnx_exports(tenant_id,id),
  CHECK((runtime='onnx')=(onnx_export_id IS NOT NULL)), CHECK((status='VALIDATING')=(validated_at IS NULL))
);
ALTER TABLE ml_deployments ADD FOREIGN KEY(tenant_id,id,active_revision) REFERENCES ml_deployment_revisions(tenant_id,deployment_id,revision) DEFERRABLE INITIALLY DEFERRED;
CREATE TABLE ml_deployment_events (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(), tenant_id uuid NOT NULL, deployment_id uuid NOT NULL,
  event text NOT NULL CHECK(event IN ('created','revision_added','validated','validation_failed','activated','rolled_back','deactivated','refused')),
  from_status text, to_status text, revision integer, endpoint_id uuid, actor uuid, reason text CHECK(reason IS NULL OR length(reason)<=1000),
  details jsonb NOT NULL DEFAULT '{}', created_at timestamptz NOT NULL DEFAULT clock_timestamp(), UNIQUE(tenant_id,id),
  FOREIGN KEY(tenant_id,deployment_id) REFERENCES ml_deployments(tenant_id,id)
);
CREATE INDEX ml_deployment_events_idx ON ml_deployment_events(tenant_id,deployment_id,created_at);

CREATE FUNCTION ml_deployment_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF (NEW.id,NEW.tenant_id,NEW.name,NEW.created_by,NEW.created_at) IS DISTINCT FROM (OLD.id,OLD.tenant_id,OLD.name,OLD.created_by,OLD.created_at) THEN RAISE EXCEPTION 'Deployment identity is immutable'; END IF;
  IF NEW.status<>OLD.status AND NOT (CASE OLD.status
    WHEN 'CREATED' THEN NEW.status='VALIDATING'
    WHEN 'VALIDATING' THEN NEW.status IN ('READY','FAILED')
    WHEN 'READY' THEN NEW.status IN ('ACTIVE','VALIDATING')
    WHEN 'ACTIVE' THEN NEW.status='INACTIVE'
    WHEN 'INACTIVE' THEN NEW.status IN ('ACTIVE','VALIDATING')
    WHEN 'FAILED' THEN NEW.status='VALIDATING' ELSE false END) THEN
    RAISE EXCEPTION 'invalid deployment transition: % -> %',OLD.status,NEW.status USING ERRCODE='check_violation';
  END IF;
  -- An active endpoint must be ACTIVE, serve the active revision's model version and runtime, and belong to nothing else.
  IF NEW.active_endpoint_id IS NOT NULL AND NEW.active_endpoint_id IS DISTINCT FROM OLD.active_endpoint_id AND NOT EXISTS(
    SELECT 1 FROM ml_endpoints e JOIN ml_deployment_revisions r ON r.tenant_id=e.tenant_id AND r.deployment_id=NEW.id AND r.revision=NEW.active_revision
    WHERE e.tenant_id=NEW.tenant_id AND e.id=NEW.active_endpoint_id AND e.status='ACTIVE' AND r.status='READY'
      AND e.model_version_id=r.model_version_id AND e.runtime=r.runtime AND e.onnx_export_id IS NOT DISTINCT FROM r.onnx_export_id) THEN
    RAISE EXCEPTION 'Deployment can only activate an ACTIVE endpoint serving a READY revision' USING ERRCODE='check_violation';
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER ml_deployment_update BEFORE UPDATE ON ml_deployments FOR EACH ROW EXECUTE FUNCTION ml_deployment_guard();
CREATE FUNCTION ml_deployment_insert_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF NEW.status<>'CREATED' OR NEW.active_endpoint_id IS NOT NULL THEN RAISE EXCEPTION 'Deployments start CREATED' USING ERRCODE='check_violation'; END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER ml_deployment_insert BEFORE INSERT ON ml_deployments FOR EACH ROW EXECUTE FUNCTION ml_deployment_insert_guard();
CREATE FUNCTION ml_revision_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF TG_OP='INSERT' THEN
    IF NEW.status<>'VALIDATING' THEN RAISE EXCEPTION 'Revisions start VALIDATING' USING ERRCODE='check_violation'; END IF;
    RETURN NEW;
  END IF;
  IF (NEW.tenant_id,NEW.deployment_id,NEW.revision,NEW.model_version_id,NEW.runtime,NEW.onnx_export_id,NEW.created_by,NEW.created_at) IS DISTINCT FROM
     (OLD.tenant_id,OLD.deployment_id,OLD.revision,OLD.model_version_id,OLD.runtime,OLD.onnx_export_id,OLD.created_by,OLD.created_at) OR OLD.status<>'VALIDATING' THEN
    RAISE EXCEPTION 'Deployment revisions are immutable once validated' USING ERRCODE='check_violation';
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER ml_revision_integrity BEFORE INSERT OR UPDATE ON ml_deployment_revisions FOR EACH ROW EXECUTE FUNCTION ml_revision_guard();

-- Disabling a deployment's active endpoint by any path (model retirement, the endpoint API) deactivates the deployment.
CREATE FUNCTION ml_endpoint_disabled_sync() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF OLD.status='ACTIVE' AND NEW.status='DISABLED' THEN
    INSERT INTO ml_deployment_events(tenant_id,deployment_id,event,from_status,to_status,revision,endpoint_id,actor,reason)
      SELECT d.tenant_id,d.id,'deactivated','ACTIVE','INACTIVE',d.active_revision,NEW.id,nullif(current_setting('app.ml_actor',true),'')::uuid,COALESCE(nullif(current_setting('app.ml_reason',true),''),'endpoint disabled')
      FROM ml_deployments d WHERE d.tenant_id=NEW.tenant_id AND d.active_endpoint_id=NEW.id;
    UPDATE ml_deployments SET status='INACTIVE',active_revision=NULL,active_endpoint_id=NULL,updated_at=now() WHERE tenant_id=NEW.tenant_id AND active_endpoint_id=NEW.id;
  END IF;
  RETURN NULL;
END $$;
CREATE TRIGGER ml_endpoint_disabled AFTER UPDATE OF status ON ml_endpoints FOR EACH ROW EXECUTE FUNCTION ml_endpoint_disabled_sync();

DO $$ DECLARE t text; BEGIN
  FOREACH t IN ARRAY ARRAY['ml_deployments','ml_deployment_revisions','ml_deployment_events'] LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY',t);
    EXECUTE format('CREATE POLICY tenant_isolation ON %I TO durable_agent_api USING (tenant_id = nullif(current_setting(''app.tenant_id'',true),'''')::uuid) WITH CHECK (tenant_id = nullif(current_setting(''app.tenant_id'',true),'''')::uuid)',t);
    EXECUTE format('GRANT SELECT,INSERT ON %I TO durable_agent_api',t);
  END LOOP;
END $$;
GRANT UPDATE ON ml_deployments,ml_deployment_revisions TO durable_agent_api;
CREATE TRIGGER ml_append_only BEFORE UPDATE OR DELETE ON ml_deployment_events FOR EACH ROW EXECUTE FUNCTION reject_evaluation_evidence_update();

INSERT INTO schema_migrations(version) VALUES('013_ml_deployments');
COMMIT;

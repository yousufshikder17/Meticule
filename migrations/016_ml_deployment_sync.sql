BEGIN;
SET LOCAL search_path=public;

-- A deployment deactivated because its endpoint was disabled (model retirement, the endpoint API) is now also recorded in
-- the served model version's lifecycle, like a deactivation through the deployment API.
CREATE OR REPLACE FUNCTION ml_endpoint_disabled_sync() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF OLD.status='ACTIVE' AND NEW.status='DISABLED' THEN
    INSERT INTO ml_deployment_events(tenant_id,deployment_id,event,from_status,to_status,revision,endpoint_id,actor,reason)
      SELECT d.tenant_id,d.id,'deactivated','ACTIVE','INACTIVE',d.active_revision,NEW.id,nullif(current_setting('app.ml_actor',true),'')::uuid,COALESCE(nullif(current_setting('app.ml_reason',true),''),'endpoint disabled')
      FROM ml_deployments d WHERE d.tenant_id=NEW.tenant_id AND d.active_endpoint_id=NEW.id;
    INSERT INTO ml_model_events(tenant_id,model_version_id,event,actor,reason,details)
      SELECT d.tenant_id,NEW.model_version_id,'deployment_changed',nullif(current_setting('app.ml_actor',true),'')::uuid,COALESCE(nullif(current_setting('app.ml_reason',true),''),'endpoint disabled'),
        jsonb_build_object('deploymentId',d.id,'revision',d.active_revision,'endpointId',NEW.id,'action','deactivated')
      FROM ml_deployments d WHERE d.tenant_id=NEW.tenant_id AND d.active_endpoint_id=NEW.id;
    UPDATE ml_deployments SET status='INACTIVE',active_revision=NULL,active_endpoint_id=NULL,updated_at=now() WHERE tenant_id=NEW.tenant_id AND active_endpoint_id=NEW.id;
  END IF;
  RETURN NULL;
END $$;

INSERT INTO schema_migrations(version) VALUES('016_ml_deployment_sync');
COMMIT;

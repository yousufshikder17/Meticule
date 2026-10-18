BEGIN;
SET LOCAL search_path=public;

-- Who registered a version (NULL for versions registered before Phase 5). Needed for separation of duties.
ALTER TABLE ml_model_versions ADD COLUMN registered_by uuid;
-- Public: versions registered before this migration recover their registrant from the append-only audit record that
-- registration has always written (one ml.model_registered event per version, by the registering user), so separation of
-- duties applies to them too. Versions without such a record keep NULL and can never be approved (independence cannot be
-- shown); register them again from a new attempt.
UPDATE ml_model_versions m SET registered_by=a.actor_id::uuid
FROM (SELECT DISTINCT ON (tenant_id,details->>'modelVersionId') tenant_id,details->>'modelVersionId' AS model_version_id,actor_id
      FROM audit_events WHERE event_type='ml.model_registered' AND actor_type='user'
        AND actor_id ~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
      ORDER BY tenant_id,details->>'modelVersionId',created_at) a
WHERE a.tenant_id=m.tenant_id AND a.model_version_id=m.id::text AND m.registered_by IS NULL;
CREATE FUNCTION ml_model_registrant_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF NEW.registered_by IS DISTINCT FROM OLD.registered_by THEN RAISE EXCEPTION 'Model provenance is immutable' USING ERRCODE='check_violation'; END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER ml_model_registrant BEFORE UPDATE ON ml_model_versions FOR EACH ROW EXECUTE FUNCTION ml_model_registrant_guard();

-- Promotion policy per registry entry. Append-only versions; the newest applies, and every decision records which one.
CREATE TABLE ml_registry_policies (
  tenant_id uuid NOT NULL, registry_entry_id uuid NOT NULL, version integer NOT NULL CHECK(version>0), policy jsonb NOT NULL,
  created_by uuid NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY(tenant_id,registry_entry_id,version), FOREIGN KEY(tenant_id,registry_entry_id) REFERENCES ml_registry_entries(tenant_id,id)
);

-- Approval decisions. The registrant can never approve their own version (separation of duties, enforced here). Public
-- tenant admins can add memberships, including service identities; an approval is a person's decision, so a service
-- identity can never approve. Identity is the authenticated subject, never a role, so changing roles changes nothing.
CREATE TABLE ml_model_approvals (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(), tenant_id uuid NOT NULL, model_version_id uuid NOT NULL,
  decision text NOT NULL CHECK(decision IN ('APPROVED','REJECTED')), approver uuid NOT NULL, note text NOT NULL DEFAULT '' CHECK(length(note)<=1000),
  created_at timestamptz NOT NULL DEFAULT now(), UNIQUE(tenant_id,id), FOREIGN KEY(tenant_id,model_version_id) REFERENCES ml_model_versions(tenant_id,id)
);
CREATE INDEX ml_model_approvals_version_idx ON ml_model_approvals(tenant_id,model_version_id,created_at);
CREATE FUNCTION ml_approval_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF EXISTS(SELECT 1 FROM ml_model_versions m WHERE m.tenant_id=NEW.tenant_id AND m.id=NEW.model_version_id AND (m.registered_by IS NULL OR m.registered_by=NEW.approver OR m.status<>'REGISTERED')) THEN
    RAISE EXCEPTION 'Approvals need a REGISTERED version with a recorded registrant and an approver other than its registrant' USING ERRCODE='check_violation';
  END IF;
  IF EXISTS(SELECT 1 FROM tenant_memberships t WHERE t.tenant_id=NEW.tenant_id AND t.identity_id=NEW.approver AND t.identity_type<>'user') THEN
    RAISE EXCEPTION 'Approvals need a user identity, not a service identity' USING ERRCODE='check_violation';
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER ml_approval_insert BEFORE INSERT ON ml_model_approvals FOR EACH ROW EXECUTE FUNCTION ml_approval_guard();

-- Immutable lifecycle history. Registration and every status change are written by triggers, so no code path can change
-- a version without a record; services add the actor and reason through transaction-local settings, and refused
-- transitions and approvals are recorded by the service with their reason codes.
CREATE TABLE ml_model_events (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(), tenant_id uuid NOT NULL, model_version_id uuid NOT NULL,
  event text NOT NULL CHECK(event IN ('registered','transitioned','transition_refused','approved','approval_rejected','deployment_changed')),
  from_status text, to_status text, actor uuid, reason text CHECK(reason IS NULL OR length(reason)<=1000), details jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT clock_timestamp(), UNIQUE(tenant_id,id),
  FOREIGN KEY(tenant_id,model_version_id) REFERENCES ml_model_versions(tenant_id,id)
);
CREATE INDEX ml_model_events_version_idx ON ml_model_events(tenant_id,model_version_id,created_at);
CREATE FUNCTION ml_model_event_capture() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF TG_OP='INSERT' THEN
    INSERT INTO ml_model_events(tenant_id,model_version_id,event,to_status,actor) VALUES(NEW.tenant_id,NEW.id,'registered',NEW.status,NEW.registered_by);
  ELSIF NEW.status IS DISTINCT FROM OLD.status THEN
    INSERT INTO ml_model_events(tenant_id,model_version_id,event,from_status,to_status,actor,reason,details)
    VALUES(NEW.tenant_id,NEW.id,'transitioned',OLD.status,NEW.status,nullif(current_setting('app.ml_actor',true),'')::uuid,nullif(current_setting('app.ml_reason',true),''),
      COALESCE(nullif(current_setting('app.ml_details',true),'')::jsonb,'{}'));
  END IF;
  RETURN NULL;
END $$;
CREATE TRIGGER ml_model_events_capture AFTER INSERT OR UPDATE OF status ON ml_model_versions FOR EACH ROW EXECUTE FUNCTION ml_model_event_capture();

DO $$ DECLARE t text; BEGIN
  FOREACH t IN ARRAY ARRAY['ml_registry_policies','ml_model_approvals','ml_model_events'] LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY',t);
    EXECUTE format('CREATE POLICY tenant_isolation ON %I TO durable_agent_api USING (tenant_id = nullif(current_setting(''app.tenant_id'',true),'''')::uuid) WITH CHECK (tenant_id = nullif(current_setting(''app.tenant_id'',true),'''')::uuid)',t);
    EXECUTE format('GRANT SELECT,INSERT ON %I TO durable_agent_api',t);
    EXECUTE format('CREATE TRIGGER ml_append_only BEFORE UPDATE OR DELETE ON %I FOR EACH ROW EXECUTE FUNCTION reject_evaluation_evidence_update()',t);
  END LOOP;
END $$;

INSERT INTO schema_migrations(version) VALUES('011_ml_governance');
COMMIT;

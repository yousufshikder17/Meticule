BEGIN;
SET LOCAL search_path=public;

-- Phase 5 experiment tracking. Additive: existing experiments keep NULL/empty metadata and their identity.
ALTER TABLE ml_experiments ADD COLUMN description text NOT NULL DEFAULT '' CHECK(length(description) <= 2000);
ALTER TABLE ml_experiments ADD COLUMN project text CHECK(project IS NULL OR length(project) BETWEEN 1 AND 120);
ALTER TABLE ml_experiments ADD COLUMN tags jsonb NOT NULL DEFAULT '{}' CHECK(jsonb_typeof(tags)='object');
ALTER TABLE ml_experiments ADD COLUMN parent_experiment_id uuid;
ALTER TABLE ml_experiments ADD FOREIGN KEY(tenant_id,parent_experiment_id) REFERENCES ml_experiments(tenant_id,id);
CREATE INDEX ml_experiments_project_idx ON ml_experiments(tenant_id,project,created_at DESC);
CREATE INDEX ml_experiments_tags_idx ON ml_experiments USING gin(tags jsonb_path_ops);

-- Only description and tags are editable. The parent is fixed at insert, so the experiment tree cannot form a cycle.
CREATE FUNCTION ml_experiment_update_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF (NEW.id,NEW.tenant_id,NEW.name,NEW.project,NEW.parent_experiment_id,NEW.created_by,NEW.created_at)
     IS DISTINCT FROM (OLD.id,OLD.tenant_id,OLD.name,OLD.project,OLD.parent_experiment_id,OLD.created_by,OLD.created_at) THEN
    RAISE EXCEPTION 'Experiment identity is immutable' USING ERRCODE='check_violation';
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER ml_experiment_update BEFORE UPDATE ON ml_experiments FOR EACH ROW EXECUTE FUNCTION ml_experiment_update_guard();
GRANT UPDATE(description,tags) ON ml_experiments TO durable_agent_api;

-- A reproduction is an ordinary job that records which job it re-runs. Set at insert and never changed.
ALTER TABLE ml_training_jobs ADD COLUMN reproduces_job_id uuid;
ALTER TABLE ml_training_jobs ADD FOREIGN KEY(tenant_id,reproduces_job_id) REFERENCES ml_training_jobs(tenant_id,id);
CREATE FUNCTION ml_job_reproduction_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF NEW.reproduces_job_id IS DISTINCT FROM OLD.reproduces_job_id THEN RAISE EXCEPTION 'ML job identity is immutable' USING ERRCODE='check_violation'; END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER ml_job_reproduction BEFORE UPDATE ON ml_training_jobs FOR EACH ROW EXECUTE FUNCTION ml_job_reproduction_guard();
CREATE INDEX ml_training_jobs_experiment_idx ON ml_training_jobs(tenant_id,experiment_id,created_at DESC);
CREATE INDEX ml_training_jobs_dataset_idx ON ml_training_jobs(tenant_id,dataset_version_id);

INSERT INTO schema_migrations(version) VALUES('008_ml_tracking');
COMMIT;

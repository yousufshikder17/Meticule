BEGIN;
SET LOCAL search_path=public;

-- Optuna studies are a fourth orchestration kind. Trials are ordinary members (one durable training job each), appended
-- as Optuna asks for them; a trial's state and objective value are always derived from its job, never stored twice.
ALTER TABLE ml_orchestrations DROP CONSTRAINT ml_orchestrations_kind_check;
ALTER TABLE ml_orchestrations ADD CONSTRAINT ml_orchestrations_kind_check CHECK(kind IN ('benchmark','search','automl','optuna'));
CREATE OR REPLACE FUNCTION ml_member_insert_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF NOT EXISTS(SELECT 1 FROM ml_orchestrations o JOIN ml_training_jobs j ON j.tenant_id=o.tenant_id AND j.experiment_id=o.experiment_id
      WHERE o.tenant_id=NEW.tenant_id AND o.id=NEW.orchestration_id AND o.kind IN ('benchmark','search','optuna') AND o.status='RUNNING' AND o.cancel_requested_at IS NULL AND j.id=NEW.job_id) THEN
    RAISE EXCEPTION 'Candidates join a running benchmark, search or study with a job from the same experiment' USING ERRCODE='check_violation';
  END IF;
  RETURN NEW;
END $$;

INSERT INTO schema_migrations(version) VALUES('010_ml_optuna');
COMMIT;

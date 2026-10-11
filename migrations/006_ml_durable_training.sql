BEGIN;
SET LOCAL search_path=public;

-- Durable deep-learning progress (Phase 3B). Additive only: existing jobs keep NULLs.
ALTER TABLE ml_training_runs ADD COLUMN resumed_from_epoch integer CHECK(resumed_from_epoch>0);
ALTER TABLE ml_training_runs ADD COLUMN best_epoch integer CHECK(best_epoch>0);
ALTER TABLE ml_training_runs ADD COLUMN stop_reason text CHECK(stop_reason IN ('completed','early_stopping'));

-- Bounded epoch summaries: one row per completed epoch of an attempt. No per-batch or tensor data.
CREATE TABLE ml_training_epochs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(), tenant_id uuid NOT NULL, training_run_id uuid NOT NULL,
  epoch integer NOT NULL CHECK(epoch BETWEEN 1 AND 1000),
  train_loss double precision NOT NULL CHECK(train_loss > '-Infinity'::float8 AND train_loss < 'Infinity'::float8),
  validation_loss double precision CHECK(validation_loss IS NULL OR (validation_loss > '-Infinity'::float8 AND validation_loss < 'Infinity'::float8)),
  elapsed_seconds double precision NOT NULL CHECK(elapsed_seconds >= 0 AND elapsed_seconds < 'Infinity'::float8),
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(tenant_id,training_run_id,epoch),
  FOREIGN KEY(tenant_id,training_run_id) REFERENCES ml_training_runs(tenant_id,id)
);

-- Checkpoints are content-addressed artifacts. The row only references bytes in the artifact store.
CREATE TABLE ml_training_checkpoints (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(), tenant_id uuid NOT NULL, training_run_id uuid NOT NULL,
  epoch integer NOT NULL CHECK(epoch BETWEEN 1 AND 1000),
  content jsonb NOT NULL, format text NOT NULL, schema_version integer NOT NULL CHECK(schema_version > 0),
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(tenant_id,training_run_id,epoch),
  FOREIGN KEY(tenant_id,training_run_id) REFERENCES ml_training_runs(tenant_id,id)
);
CREATE INDEX ml_training_checkpoints_run_idx ON ml_training_checkpoints(tenant_id,training_run_id,epoch);

-- Progress is accepted only while its attempt is the current TRAINING attempt. Fencing by lease happens in the
-- application transaction (ownedRun); this trigger prevents writes that bypass it after the attempt has moved on.
CREATE FUNCTION ml_progress_insert_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF NOT EXISTS(SELECT 1 FROM ml_training_runs t WHERE t.tenant_id=NEW.tenant_id AND t.id=NEW.training_run_id AND t.status='TRAINING') THEN
    RAISE EXCEPTION 'ML progress requires a training attempt' USING ERRCODE='check_violation';
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER ml_epoch_insert BEFORE INSERT ON ml_training_epochs FOR EACH ROW EXECUTE FUNCTION ml_progress_insert_guard();
CREATE TRIGGER ml_checkpoint_insert BEFORE INSERT ON ml_training_checkpoints FOR EACH ROW EXECUTE FUNCTION ml_progress_insert_guard();
CREATE TRIGGER ml_epoch_append_only BEFORE UPDATE OR DELETE ON ml_training_epochs FOR EACH ROW EXECUTE FUNCTION reject_evaluation_evidence_update();
CREATE TRIGGER ml_checkpoint_append_only BEFORE UPDATE OR DELETE ON ml_training_checkpoints FOR EACH ROW EXECUTE FUNCTION reject_evaluation_evidence_update();

DO $$ DECLARE t text; BEGIN
  FOREACH t IN ARRAY ARRAY['ml_training_epochs','ml_training_checkpoints'] LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY',t);
    EXECUTE format('CREATE POLICY tenant_isolation ON %I TO durable_agent_api USING (tenant_id = nullif(current_setting(''app.tenant_id'',true),'''')::uuid) WITH CHECK (tenant_id = nullif(current_setting(''app.tenant_id'',true),'''')::uuid)',t);
    EXECUTE format('GRANT SELECT,INSERT ON %I TO durable_agent_api',t);
  END LOOP;
END $$;

INSERT INTO schema_migrations(version) VALUES('006_ml_durable_training');
COMMIT;

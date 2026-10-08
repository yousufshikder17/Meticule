BEGIN;
SET LOCAL search_path=public;

-- Cross-validation evidence: aggregate ("cv", with std) and per-fold ("cv_fold") scores beside the Phase 1 partitions.
ALTER TABLE ml_metrics ADD COLUMN fold integer CHECK(fold>=0);
ALTER TABLE ml_metrics ADD COLUMN std double precision CHECK(std>=0 AND std<'Infinity'::float8);
ALTER TABLE ml_metrics DROP CONSTRAINT ml_metrics_partition_check;
ALTER TABLE ml_metrics DROP CONSTRAINT ml_metrics_tenant_id_training_run_id_name_partition_key;
ALTER TABLE ml_metrics ADD CONSTRAINT ml_metrics_partition_check CHECK(partition IN ('train','validation','test','cv','cv_fold'));
ALTER TABLE ml_metrics ADD CONSTRAINT ml_metrics_fold_shape CHECK((partition='cv_fold')=(fold IS NOT NULL));
-- A metric is unique per attempt, name, partition and (for cv_fold) fold.
CREATE UNIQUE INDEX ml_metrics_identity_idx ON ml_metrics(tenant_id,training_run_id,name,partition,COALESCE(fold,-1));

-- Measured seconds reported by the backend (fit, predict, CV fit). Written only while the attempt is non-terminal.
ALTER TABLE ml_training_runs ADD COLUMN performance jsonb NOT NULL DEFAULT '{}';

INSERT INTO schema_migrations(version) VALUES('004_ml_cross_validation');
COMMIT;

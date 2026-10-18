BEGIN;
SET LOCAL search_path=public;

-- Rendered reports: the request, the records it was computed from, the computed numbers and the image in the artifact
-- store. Append-only; a report is never re-rendered in place.
CREATE TABLE ml_reports (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(), tenant_id uuid NOT NULL,
  kind text NOT NULL CHECK(kind IN ('training_curves','metric_comparison','optimization_history','confusion_matrix','residuals','roc_curve','precision_recall','forecast','anomaly_scores','drift','drift_trend')),
  format text NOT NULL CHECK(format IN ('png','svg')), spec jsonb NOT NULL, sources jsonb NOT NULL, summary jsonb NOT NULL,
  content jsonb NOT NULL, producer jsonb NOT NULL, created_by uuid NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(tenant_id,id)
);
CREATE INDEX ml_reports_idx ON ml_reports(tenant_id,kind,created_at DESC);
CREATE TRIGGER ml_append_only BEFORE UPDATE OR DELETE ON ml_reports FOR EACH ROW EXECUTE FUNCTION reject_evaluation_evidence_update();
ALTER TABLE ml_reports ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON ml_reports TO durable_agent_api
  USING(tenant_id=nullif(current_setting('app.tenant_id',true),'')::uuid) WITH CHECK(tenant_id=nullif(current_setting('app.tenant_id',true),'')::uuid);
GRANT SELECT,INSERT ON ml_reports TO durable_agent_api;

INSERT INTO schema_migrations(version) VALUES('015_ml_reports');
COMMIT;

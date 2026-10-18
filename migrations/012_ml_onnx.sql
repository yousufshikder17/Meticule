BEGIN;
SET LOCAL search_path=public;

-- One row per ONNX export attempt of a model version, append-only. Only a VERIFIED export (converted, accepted by the
-- runtime's gate and within the parity tolerance against the native artifact) may back an ONNX deployment.
CREATE TABLE ml_onnx_exports (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(), tenant_id uuid NOT NULL, model_version_id uuid NOT NULL,
  status text NOT NULL CHECK(status IN ('VERIFIED','PARITY_FAILED','UNSUPPORTED')),
  content jsonb, opsets jsonb, signature jsonb, operators jsonb, producer jsonb, parity jsonb, failure jsonb,
  created_by uuid NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), UNIQUE(tenant_id,id),
  FOREIGN KEY(tenant_id,model_version_id) REFERENCES ml_model_versions(tenant_id,id),
  CHECK((status='UNSUPPORTED')=(content IS NULL)), CHECK((status='UNSUPPORTED')=(failure IS NOT NULL)), CHECK(status='UNSUPPORTED' OR parity IS NOT NULL)
);
CREATE INDEX ml_onnx_exports_version_idx ON ml_onnx_exports(tenant_id,model_version_id,created_at DESC);
CREATE TRIGGER ml_append_only BEFORE UPDATE OR DELETE ON ml_onnx_exports FOR EACH ROW EXECUTE FUNCTION reject_evaluation_evidence_update();
ALTER TABLE ml_onnx_exports ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON ml_onnx_exports TO durable_agent_api
  USING(tenant_id=nullif(current_setting('app.tenant_id',true),'')::uuid) WITH CHECK(tenant_id=nullif(current_setting('app.tenant_id',true),'')::uuid);
GRANT SELECT,INSERT ON ml_onnx_exports TO durable_agent_api;

INSERT INTO schema_migrations(version) VALUES('012_ml_onnx');
COMMIT;

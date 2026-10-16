BEGIN;
SET LOCAL search_path=public;

-- Bounded embedding indexes (Phase 4 M6). The vectors of one dataset version, embedded by one pinned READY model version
-- through one endpoint, searched exactly with pgvector (no approximate index, no separate vector store). An index is built
-- in bounded steps while BUILDING, then becomes READY (or FAILED) and is append-only from then on.
CREATE TABLE ml_embedding_indexes (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(), tenant_id uuid NOT NULL,
  name text NOT NULL CHECK(char_length(name) BETWEEN 1 AND 120),
  endpoint_id uuid NOT NULL, model_version_id uuid NOT NULL, dataset_version_id uuid NOT NULL,
  dimension integer NOT NULL CHECK(dimension BETWEEN 1 AND 4096),
  metric text NOT NULL CHECK(metric IN ('cosine','dot','euclidean')),
  row_count integer NOT NULL CHECK(row_count BETWEEN 1 AND 5000),
  payload_columns jsonb NOT NULL DEFAULT '[]'::jsonb CHECK(jsonb_typeof(payload_columns)='array' AND jsonb_array_length(payload_columns) <= 16),
  status text NOT NULL DEFAULT 'BUILDING' CHECK(status IN ('BUILDING','READY','FAILED')),
  created_by uuid NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), ready_at timestamptz,
  CHECK((status='READY') = (ready_at IS NOT NULL)),
  UNIQUE(tenant_id,name), UNIQUE(tenant_id,id), UNIQUE(tenant_id,id,dimension),
  FOREIGN KEY(tenant_id,endpoint_id) REFERENCES ml_endpoints(tenant_id,id),
  FOREIGN KEY(tenant_id,model_version_id) REFERENCES ml_model_versions(tenant_id,id),
  FOREIGN KEY(tenant_id,dataset_version_id) REFERENCES ml_dataset_versions(tenant_id,id)
);

-- One vector per dataset row. The composite key carries the dimension, so a vector of another width cannot attach.
CREATE TABLE ml_embedding_vectors (
  tenant_id uuid NOT NULL, index_id uuid NOT NULL, dimension integer NOT NULL,
  row_index integer NOT NULL CHECK(row_index BETWEEN 0 AND 4999),
  embedding vector NOT NULL CHECK(vector_dims(embedding) = dimension),
  payload jsonb NOT NULL DEFAULT '{}'::jsonb CHECK(jsonb_typeof(payload)='object'),
  PRIMARY KEY(tenant_id,index_id,row_index), -- also the (tenant_id,index_id) lookup index for search
  FOREIGN KEY(tenant_id,index_id,dimension) REFERENCES ml_embedding_indexes(tenant_id,id,dimension)
);

-- Lifecycle: BUILDING -> READY | FAILED only; identity never changes; indexes are never deleted.
CREATE FUNCTION ml_embedding_index_transition() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF TG_OP = 'DELETE' THEN
    RAISE EXCEPTION 'Embedding indexes are append-only' USING ERRCODE='check_violation';
  END IF;
  IF (NEW.id, NEW.tenant_id, NEW.name, NEW.endpoint_id, NEW.model_version_id, NEW.dataset_version_id, NEW.dimension, NEW.metric,
      NEW.row_count, NEW.payload_columns, NEW.created_by, NEW.created_at)
     IS DISTINCT FROM (OLD.id, OLD.tenant_id, OLD.name, OLD.endpoint_id, OLD.model_version_id, OLD.dataset_version_id, OLD.dimension,
      OLD.metric, OLD.row_count, OLD.payload_columns, OLD.created_by, OLD.created_at) THEN
    RAISE EXCEPTION 'Embedding index identity is immutable' USING ERRCODE='check_violation';
  END IF;
  IF NOT (OLD.status = 'BUILDING' AND NEW.status IN ('READY','FAILED')) THEN
    RAISE EXCEPTION 'Invalid embedding index transition' USING ERRCODE='check_violation';
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER ml_embedding_index_transition BEFORE UPDATE OR DELETE ON ml_embedding_indexes FOR EACH ROW EXECUTE FUNCTION ml_embedding_index_transition();

-- Vectors are accepted only while their index is BUILDING, and never change afterwards.
CREATE FUNCTION ml_embedding_vector_insert_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF NOT EXISTS(SELECT 1 FROM ml_embedding_indexes i WHERE i.tenant_id=NEW.tenant_id AND i.id=NEW.index_id AND i.status='BUILDING') THEN
    RAISE EXCEPTION 'Embedding vectors require a building index' USING ERRCODE='check_violation';
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER ml_embedding_vector_insert BEFORE INSERT ON ml_embedding_vectors FOR EACH ROW EXECUTE FUNCTION ml_embedding_vector_insert_guard();
CREATE TRIGGER ml_embedding_vector_append_only BEFORE UPDATE OR DELETE ON ml_embedding_vectors FOR EACH ROW EXECUTE FUNCTION reject_evaluation_evidence_update();

DO $$ DECLARE t text; BEGIN
  FOREACH t IN ARRAY ARRAY['ml_embedding_indexes','ml_embedding_vectors'] LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY',t);
    EXECUTE format('CREATE POLICY tenant_isolation ON %I TO durable_agent_api USING (tenant_id = nullif(current_setting(''app.tenant_id'',true),'''')::uuid) WITH CHECK (tenant_id = nullif(current_setting(''app.tenant_id'',true),'''')::uuid)',t);
    EXECUTE format('GRANT SELECT,INSERT ON %I TO durable_agent_api',t);
  END LOOP;
END $$;
-- The only update the API role may make: the lifecycle columns, which the transition trigger guards.
GRANT UPDATE (status, ready_at) ON ml_embedding_indexes TO durable_agent_api;

INSERT INTO schema_migrations(version) VALUES('007_ml_embeddings');
COMMIT;

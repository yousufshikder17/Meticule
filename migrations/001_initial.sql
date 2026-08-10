-- Initial public-release database baseline.
-- Derived from the schema produced by the verified private development migration chain.

BEGIN;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'agent_demo_reader') THEN
    CREATE ROLE agent_demo_reader NOLOGIN;
  END IF;
END $$;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'durable_agent_api') THEN
    CREATE ROLE durable_agent_api NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
  END IF;
END $$;

GRANT durable_agent_api TO CURRENT_USER;

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: pgcrypto; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA public;


--
-- Name: EXTENSION pgcrypto; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION pgcrypto IS 'cryptographic functions';


--
-- Name: vector; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;


--
-- Name: EXTENSION vector; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION vector IS 'vector data type and ivfflat and hnsw access methods';


--
-- Name: approval_decision; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.approval_decision AS ENUM (
    'pending',
    'approved',
    'rejected'
);


--
-- Name: model_attempt_status; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.model_attempt_status AS ENUM (
    'pending',
    'running',
    'succeeded',
    'failed',
    'cancelled',
    'unknown'
);


--
-- Name: run_status; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.run_status AS ENUM (
    'queued',
    'claimed',
    'running',
    'waiting_for_approval',
    'paused',
    'cancelling',
    'completed',
    'failed',
    'cancelled'
);


--
-- Name: step_status; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.step_status AS ENUM (
    'pending',
    'running',
    'succeeded',
    'failed',
    'cancelled',
    'unknown'
);


--
-- Name: enforce_orchestrated_completion(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.enforce_orchestrated_completion() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
  IF NEW.status='completed' AND OLD.status<>'completed' THEN
    IF EXISTS (
      SELECT 1 FROM run_delegations d JOIN runs child ON child.tenant_id=d.tenant_id AND child.id=d.child_run_id
      WHERE d.tenant_id=NEW.tenant_id AND d.parent_run_id=NEW.id
        AND child.status NOT IN ('completed','failed','cancelled')
    ) THEN RAISE EXCEPTION 'orchestrated run cannot complete while child runs remain active' USING ERRCODE='check_violation'; END IF;
    IF EXISTS (
      SELECT 1 FROM run_delegations d JOIN runs child ON child.tenant_id=d.tenant_id AND child.id=d.child_run_id
      WHERE d.tenant_id=NEW.tenant_id AND d.parent_run_id=NEW.id AND d.required
        AND (child.status<>'completed' OR child.final_output IS NULL OR child.final_output IN ('null'::jsonb,'{}'::jsonb,'[]'::jsonb,'""'::jsonb))
    ) THEN RAISE EXCEPTION 'orchestrated run cannot complete with failed or empty required child output' USING ERRCODE='check_violation'; END IF;
  END IF;
  RETURN NEW;
END $$;


--
-- Name: enforce_run_transition(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.enforce_run_transition() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
  IF NEW.status = OLD.status THEN RETURN NEW; END IF;
  IF NOT (CASE OLD.status
    WHEN 'queued' THEN NEW.status IN ('claimed', 'cancelling')
    WHEN 'claimed' THEN NEW.status IN ('running', 'queued', 'cancelling')
    WHEN 'running' THEN NEW.status IN ('waiting_for_approval', 'paused', 'cancelling', 'completed', 'failed')
    WHEN 'waiting_for_approval' THEN NEW.status IN ('queued', 'cancelling')
    WHEN 'paused' THEN NEW.status IN ('queued', 'cancelling')
    WHEN 'cancelling' THEN NEW.status IN ('cancelled', 'failed')
    ELSE false
  END) THEN
    RAISE EXCEPTION 'invalid run transition: % -> %', OLD.status, NEW.status USING ERRCODE = 'check_violation';
  END IF;
  RETURN NEW;
END $$;


--
-- Name: protect_evaluation_case_run_identity(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.protect_evaluation_case_run_identity() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
  IF NEW.id IS DISTINCT FROM OLD.id OR NEW.tenant_id IS DISTINCT FROM OLD.tenant_id OR NEW.execution_id IS DISTINCT FROM OLD.execution_id
     OR NEW.case_id IS DISTINCT FROM OLD.case_id OR NEW.run_id IS DISTINCT FROM OLD.run_id OR NEW.created_at IS DISTINCT FROM OLD.created_at
  THEN RAISE EXCEPTION 'evaluation case run identity is immutable'; END IF;
  RETURN NEW;
END $$;


--
-- Name: protect_evaluation_execution_identity(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.protect_evaluation_execution_identity() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
  IF NEW.id IS DISTINCT FROM OLD.id OR NEW.tenant_id IS DISTINCT FROM OLD.tenant_id OR NEW.suite_id IS DISTINCT FROM OLD.suite_id
     OR NEW.agent_id IS DISTINCT FROM OLD.agent_id OR NEW.agent_version IS DISTINCT FROM OLD.agent_version OR NEW.agent_snapshot IS DISTINCT FROM OLD.agent_snapshot
     OR NEW.agent_snapshot_hash IS DISTINCT FROM OLD.agent_snapshot_hash OR NEW.mode IS DISTINCT FROM OLD.mode OR NEW.candidate_label IS DISTINCT FROM OLD.candidate_label
     OR NEW.case_count IS DISTINCT FROM OLD.case_count OR NEW.created_by IS DISTINCT FROM OLD.created_by OR NEW.created_at IS DISTINCT FROM OLD.created_at
  THEN RAISE EXCEPTION 'evaluation execution candidate identity is immutable'; END IF;
  RETURN NEW;
END $$;


--
-- Name: protect_evaluation_suite_version(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.protect_evaluation_suite_version() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
  IF NEW.id IS DISTINCT FROM OLD.id OR NEW.tenant_id IS DISTINCT FROM OLD.tenant_id
     OR NEW.logical_id IS DISTINCT FROM OLD.logical_id OR NEW.version IS DISTINCT FROM OLD.version
     OR NEW.supersedes_id IS DISTINCT FROM OLD.supersedes_id OR NEW.created_by IS DISTINCT FROM OLD.created_by
     OR NEW.name IS DISTINCT FROM OLD.name OR NEW.description IS DISTINCT FROM OLD.description
     OR NEW.thresholds IS DISTINCT FROM OLD.thresholds OR NEW.created_at IS DISTINCT FROM OLD.created_at
     OR NOT (OLD.status='active' AND NEW.status='archived') THEN RAISE EXCEPTION 'evaluation suite versions are immutable except active-to-archived status'; END IF;
  RETURN NEW;
END $$;


--
-- Name: protect_frozen_approval_action(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.protect_frozen_approval_action() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
  IF NEW.tenant_id IS DISTINCT FROM OLD.tenant_id
     OR NEW.run_id IS DISTINCT FROM OLD.run_id
     OR NEW.step_id IS DISTINCT FROM OLD.step_id
     OR NEW.tool_name IS DISTINCT FROM OLD.tool_name
     OR NEW.validated_arguments IS DISTINCT FROM OLD.validated_arguments
     OR NEW.canonical_argument_hash IS DISTINCT FROM OLD.canonical_argument_hash
     OR NEW.idempotency_key IS DISTINCT FROM OLD.idempotency_key
     OR NEW.requester_id IS DISTINCT FROM OLD.requester_id
     OR NEW.required_approver_role IS DISTINCT FROM OLD.required_approver_role
     OR NEW.requested_roles IS DISTINCT FROM OLD.requested_roles
     OR NEW.tool_risk IS DISTINCT FROM OLD.tool_risk
     OR NEW.risk_explanation IS DISTINCT FROM OLD.risk_explanation
     OR NEW.separation_of_duties IS DISTINCT FROM OLD.separation_of_duties
     OR NEW.created_at IS DISTINCT FROM OLD.created_at THEN
    RAISE EXCEPTION 'frozen approval action is immutable' USING ERRCODE = 'check_violation';
  END IF;
  IF OLD.decision <> 'pending' AND NEW.decision <> OLD.decision THEN
    RAISE EXCEPTION 'approval decision is immutable' USING ERRCODE = 'check_violation';
  END IF;
  RETURN NEW;
END $$;


--
-- Name: reject_evaluation_evidence_update(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.reject_evaluation_evidence_update() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN RAISE EXCEPTION '% rows are append-only', TG_TABLE_NAME; END $$;


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: agents; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.agents (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    tenant_id uuid NOT NULL,
    created_by uuid NOT NULL,
    name text NOT NULL,
    system_instructions text NOT NULL,
    model_config jsonb NOT NULL,
    allowed_tools jsonb DEFAULT '[]'::jsonb NOT NULL,
    maximum_steps integer NOT NULL,
    token_budget bigint NOT NULL,
    cost_budget_microusd bigint NOT NULL,
    approval_policy jsonb DEFAULT '{}'::jsonb NOT NULL,
    output_schema jsonb,
    version integer DEFAULT 1 NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    composition_config jsonb DEFAULT '{"memory": "disabled", "planner": "disabled", "retriever": "disabled", "outputParser": "native", "contextBuilder": "native", "executionEngine": "native"}'::jsonb NOT NULL,
    context_policy jsonb DEFAULT '{"recentSteps": 20, "maxInputTokens": 8192, "summaryTargetTokens": 2000}'::jsonb NOT NULL,
    memory_policy jsonb DEFAULT '{"writeEnabled": false, "allowedScopes": [], "maxContextItems": 5, "maxWritesPerRun": 0, "maxContextTokens": 1000, "retrievalEnabled": false}'::jsonb NOT NULL,
    retrieval_policy jsonb DEFAULT '{"enabled": false, "minimumScore": 0.2, "maxContextChunks": 5, "maxContextTokens": 2000}'::jsonb NOT NULL,
    connector_policy jsonb DEFAULT '{"enabled": false, "connectorIds": [], "maxContextTools": 20}'::jsonb NOT NULL,
    skill_policy jsonb DEFAULT '{"enabled": false, "skillIds": [], "maxContextTokens": 2000}'::jsonb NOT NULL,
    orchestration_policy jsonb DEFAULT '{"enabled": false, "allowedRoles": [], "maximumDepth": 1, "allowedAgentIds": [], "maximumChildren": 0, "maximumParallel": 1, "allowSharedContext": false, "maximumChildTokenBudget": 0, "maximumChildCostBudgetMicrousd": 0}'::jsonb NOT NULL,
    CONSTRAINT agents_cost_budget_microusd_check CHECK ((cost_budget_microusd >= 0)),
    CONSTRAINT agents_maximum_steps_check CHECK ((maximum_steps > 0)),
    CONSTRAINT agents_name_check CHECK (((length(name) >= 1) AND (length(name) <= 120))),
    CONSTRAINT agents_system_instructions_check CHECK ((length(system_instructions) > 0)),
    CONSTRAINT agents_token_budget_check CHECK ((token_budget >= 0)),
    CONSTRAINT agents_version_check CHECK ((version > 0))
);


--
-- Name: approvals; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.approvals (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    tenant_id uuid NOT NULL,
    run_id uuid NOT NULL,
    step_id uuid NOT NULL,
    tool_name text NOT NULL,
    validated_arguments jsonb NOT NULL,
    risk_explanation text NOT NULL,
    requester_id uuid NOT NULL,
    required_approver_role text NOT NULL,
    decision public.approval_decision DEFAULT 'pending'::public.approval_decision NOT NULL,
    approver_id uuid,
    decided_at timestamp with time zone,
    comment text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    canonical_argument_hash text NOT NULL,
    idempotency_key text NOT NULL,
    tool_risk text NOT NULL,
    requested_roles jsonb DEFAULT '[]'::jsonb NOT NULL,
    separation_of_duties boolean DEFAULT true NOT NULL,
    consumed_at timestamp with time zone,
    consumed_by_worker text,
    CONSTRAINT approvals_argument_hash_format CHECK ((canonical_argument_hash ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT approvals_consumed_only_when_approved CHECK (((consumed_at IS NULL) OR (decision = 'approved'::public.approval_decision))),
    CONSTRAINT approvals_consumption_fields CHECK (((consumed_at IS NULL) = (consumed_by_worker IS NULL))),
    CONSTRAINT approvals_requested_roles_array CHECK ((jsonb_typeof(requested_roles) = 'array'::text)),
    CONSTRAINT approvals_tool_risk CHECK ((tool_risk = ANY (ARRAY['low'::text, 'medium'::text, 'high'::text]))),
    CONSTRAINT decision_fields CHECK (((decision = 'pending'::public.approval_decision) = ((approver_id IS NULL) AND (decided_at IS NULL))))
);


--
-- Name: audit_events; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.audit_events (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    tenant_id uuid NOT NULL,
    run_id uuid,
    actor_type text NOT NULL,
    actor_id text NOT NULL,
    event_type text NOT NULL,
    details jsonb DEFAULT '{}'::jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: checkpoints; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.checkpoints (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    tenant_id uuid NOT NULL,
    run_id uuid NOT NULL,
    step_id uuid,
    state jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    version integer NOT NULL,
    kind text DEFAULT 'working_state'::text NOT NULL,
    state_checksum text NOT NULL,
    provenance jsonb DEFAULT '{}'::jsonb NOT NULL
);


--
-- Name: connector_invocations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.connector_invocations (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    tenant_id uuid NOT NULL,
    connector_id uuid NOT NULL,
    run_id uuid NOT NULL,
    tool_execution_id uuid,
    idempotency_key text NOT NULL,
    request_id text NOT NULL,
    tool_name text NOT NULL,
    status text NOT NULL,
    duration_ms integer,
    error_code text,
    result_metadata jsonb,
    started_at timestamp with time zone,
    completed_at timestamp with time zone,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT connector_invocations_duration_ms_check CHECK (((duration_ms IS NULL) OR (duration_ms >= 0))),
    CONSTRAINT connector_invocations_status_check CHECK ((status = ANY (ARRAY['pending'::text, 'running'::text, 'succeeded'::text, 'failed'::text, 'unknown'::text])))
);


--
-- Name: connector_rate_limits; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.connector_rate_limits (
    tenant_id uuid NOT NULL,
    connector_id uuid NOT NULL,
    window_start timestamp with time zone NOT NULL,
    request_count integer NOT NULL,
    CONSTRAINT connector_rate_limits_request_count_check CHECK ((request_count > 0))
);


--
-- Name: connector_tools; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.connector_tools (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    tenant_id uuid NOT NULL,
    connector_id uuid NOT NULL,
    name text NOT NULL,
    title text,
    description text,
    input_schema jsonb NOT NULL,
    output_schema jsonb,
    annotations jsonb DEFAULT '{}'::jsonb NOT NULL,
    schema_hash text NOT NULL,
    enabled boolean DEFAULT true NOT NULL,
    discovered_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT connector_tools_description_check CHECK (((description IS NULL) OR (length(description) <= 10000))),
    CONSTRAINT connector_tools_name_check CHECK (((length(name) >= 1) AND (length(name) <= 200))),
    CONSTRAINT connector_tools_schema_hash_check CHECK ((schema_hash ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT connector_tools_title_check CHECK (((title IS NULL) OR (length(title) <= 500)))
);


--
-- Name: connectors; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.connectors (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    tenant_id uuid NOT NULL,
    name text NOT NULL,
    transport text NOT NULL,
    endpoint_url text NOT NULL,
    credential_ref text,
    allowed_tools jsonb DEFAULT '[]'::jsonb NOT NULL,
    required_roles jsonb DEFAULT '[]'::jsonb NOT NULL,
    rate_limit_per_minute integer NOT NULL,
    protocol_version text DEFAULT '2026-07-28'::text NOT NULL,
    status text DEFAULT 'active'::text NOT NULL,
    server_capabilities jsonb,
    server_info jsonb,
    server_instructions text,
    last_health_status text,
    last_health_at timestamp with time zone,
    last_discovered_at timestamp with time zone,
    created_by uuid NOT NULL,
    revoked_at timestamp with time zone,
    revoked_by uuid,
    revocation_reason text,
    version integer DEFAULT 1 NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT connectors_credential_ref_check CHECK (((credential_ref IS NULL) OR (credential_ref ~ '^CONNECTOR_SECRET_[A-Z0-9_]+$'::text))),
    CONSTRAINT connectors_endpoint_url_check CHECK (((length(endpoint_url) >= 1) AND (length(endpoint_url) <= 2000))),
    CONSTRAINT connectors_last_health_status_check CHECK (((last_health_status IS NULL) OR (last_health_status = ANY (ARRAY['healthy'::text, 'unhealthy'::text])))),
    CONSTRAINT connectors_name_check CHECK (((length(name) >= 1) AND (length(name) <= 200))),
    CONSTRAINT connectors_rate_limit_per_minute_check CHECK (((rate_limit_per_minute >= 1) AND (rate_limit_per_minute <= 10000))),
    CONSTRAINT connectors_revocation_fields CHECK ((((status = 'revoked'::text) AND (revoked_at IS NOT NULL) AND (revoked_by IS NOT NULL) AND (revocation_reason IS NOT NULL)) OR ((status <> 'revoked'::text) AND (revoked_at IS NULL) AND (revoked_by IS NULL) AND (revocation_reason IS NULL)))),
    CONSTRAINT connectors_server_instructions_check CHECK (((server_instructions IS NULL) OR (length(server_instructions) <= 10000))),
    CONSTRAINT connectors_status_check CHECK ((status = ANY (ARRAY['active'::text, 'degraded'::text, 'revoked'::text]))),
    CONSTRAINT connectors_transport_check CHECK ((transport = 'mcp_streamable_http'::text)),
    CONSTRAINT connectors_version_check CHECK ((version > 0))
);


--
-- Name: context_builds; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.context_builds (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    tenant_id uuid NOT NULL,
    run_id uuid NOT NULL,
    checkpoint_id uuid NOT NULL,
    plan_id uuid,
    summary_id uuid,
    token_budget integer NOT NULL,
    token_estimate integer NOT NULL,
    included_step_sequences jsonb NOT NULL,
    omitted_step_sequences jsonb NOT NULL,
    provenance jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    selected_memory_ids jsonb DEFAULT '[]'::jsonb NOT NULL,
    selected_document_chunk_ids jsonb DEFAULT '[]'::jsonb NOT NULL,
    selected_skill_ids jsonb DEFAULT '[]'::jsonb NOT NULL,
    selected_connector_tool_ids jsonb DEFAULT '[]'::jsonb NOT NULL,
    selected_child_run_ids jsonb DEFAULT '[]'::jsonb NOT NULL,
    CONSTRAINT context_builds_token_budget_check CHECK ((token_budget > 0)),
    CONSTRAINT context_builds_token_estimate_check CHECK ((token_estimate >= 0))
);


--
-- Name: context_summaries; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.context_summaries (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    tenant_id uuid NOT NULL,
    run_id uuid NOT NULL,
    through_sequence integer NOT NULL,
    source_step_count integer NOT NULL,
    content text NOT NULL,
    token_estimate integer NOT NULL,
    provenance jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT context_summaries_source_step_count_check CHECK ((source_step_count > 0)),
    CONSTRAINT context_summaries_through_sequence_check CHECK ((through_sequence > 0)),
    CONSTRAINT context_summaries_token_estimate_check CHECK ((token_estimate >= 0))
);


--
-- Name: demo_products; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.demo_products (
    id integer NOT NULL,
    name text NOT NULL,
    category text NOT NULL,
    price_cents integer NOT NULL,
    CONSTRAINT demo_products_price_cents_check CHECK ((price_cents >= 0))
);


--
-- Name: document_chunks; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.document_chunks (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    tenant_id uuid NOT NULL,
    document_id uuid NOT NULL,
    chunk_index integer NOT NULL,
    content_text text NOT NULL,
    token_estimate integer NOT NULL,
    content_hash text NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb NOT NULL,
    embedding public.vector NOT NULL,
    embedding_provider text NOT NULL,
    embedding_model text NOT NULL,
    embedding_dimension integer NOT NULL,
    deleted_at timestamp with time zone,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT document_chunks_chunk_index_check CHECK ((chunk_index >= 0)),
    CONSTRAINT document_chunks_content_hash_check CHECK ((content_hash ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT document_chunks_content_text_check CHECK (((length(content_text) >= 1) AND (length(content_text) <= 10000))),
    CONSTRAINT document_chunks_embedding_dimension_check CHECK (((embedding_dimension >= 1) AND (embedding_dimension <= 4096))),
    CONSTRAINT document_chunks_token_estimate_check CHECK ((token_estimate > 0))
);


--
-- Name: documents; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.documents (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    tenant_id uuid NOT NULL,
    logical_id uuid DEFAULT gen_random_uuid() NOT NULL,
    version integer NOT NULL,
    owner_user_id uuid NOT NULL,
    visibility text NOT NULL,
    title text NOT NULL,
    media_type text NOT NULL,
    source_uri text,
    content_text text NOT NULL,
    content_hash text NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb NOT NULL,
    status text NOT NULL,
    embedding_provider text NOT NULL,
    embedding_model text NOT NULL,
    embedding_dimension integer NOT NULL,
    supersedes_id uuid,
    is_current boolean DEFAULT true NOT NULL,
    error_details jsonb,
    deleted_at timestamp with time zone,
    deleted_by uuid,
    deletion_reason text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT documents_content_hash_check CHECK ((content_hash ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT documents_content_text_check CHECK (((length(content_text) >= 1) AND (length(content_text) <= 2000000))),
    CONSTRAINT documents_deletion_fields CHECK ((((status = 'deleted'::text) AND (deleted_at IS NOT NULL) AND (deleted_by IS NOT NULL) AND (deletion_reason IS NOT NULL) AND (is_current = false)) OR ((status <> 'deleted'::text) AND (deleted_at IS NULL) AND (deleted_by IS NULL) AND (deletion_reason IS NULL)))),
    CONSTRAINT documents_embedding_dimension_check CHECK (((embedding_dimension >= 1) AND (embedding_dimension <= 4096))),
    CONSTRAINT documents_media_type_check CHECK ((media_type = ANY (ARRAY['text/plain'::text, 'text/markdown'::text, 'application/json'::text]))),
    CONSTRAINT documents_source_uri_check CHECK (((source_uri IS NULL) OR (length(source_uri) <= 2000))),
    CONSTRAINT documents_status_check CHECK ((status = ANY (ARRAY['pending'::text, 'ready'::text, 'failed'::text, 'deleted'::text]))),
    CONSTRAINT documents_title_check CHECK (((length(title) >= 1) AND (length(title) <= 500))),
    CONSTRAINT documents_version_check CHECK ((version > 0)),
    CONSTRAINT documents_visibility_check CHECK ((visibility = ANY (ARRAY['private'::text, 'tenant'::text])))
);


--
-- Name: evaluation_case_runs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.evaluation_case_runs (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    tenant_id uuid NOT NULL,
    execution_id uuid NOT NULL,
    case_id uuid NOT NULL,
    run_id uuid NOT NULL,
    status text NOT NULL,
    metrics jsonb,
    failures jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    started_at timestamp with time zone,
    completed_at timestamp with time zone,
    CONSTRAINT evaluation_case_runs_status_check CHECK ((status = ANY (ARRAY['pending'::text, 'scoring'::text, 'passed'::text, 'failed'::text, 'error'::text]))),
    CONSTRAINT evaluation_case_runs_terminal CHECK (((status = ANY (ARRAY['passed'::text, 'failed'::text, 'error'::text])) = (completed_at IS NOT NULL)))
);


--
-- Name: evaluation_cases; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.evaluation_cases (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    tenant_id uuid NOT NULL,
    suite_id uuid NOT NULL,
    ordinal integer NOT NULL,
    name text NOT NULL,
    goal text NOT NULL,
    expectations jsonb NOT NULL,
    tags jsonb DEFAULT '[]'::jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT evaluation_cases_goal_check CHECK (((length(goal) >= 1) AND (length(goal) <= 100000))),
    CONSTRAINT evaluation_cases_name_check CHECK (((length(name) >= 1) AND (length(name) <= 160))),
    CONSTRAINT evaluation_cases_ordinal_check CHECK ((ordinal > 0))
);


--
-- Name: evaluation_executions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.evaluation_executions (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    tenant_id uuid NOT NULL,
    suite_id uuid NOT NULL,
    agent_id uuid NOT NULL,
    agent_version integer NOT NULL,
    agent_snapshot jsonb NOT NULL,
    agent_snapshot_hash text NOT NULL,
    mode text NOT NULL,
    candidate_label text NOT NULL,
    status text NOT NULL,
    case_count integer NOT NULL,
    passed_count integer DEFAULT 0 NOT NULL,
    failed_count integer DEFAULT 0 NOT NULL,
    gate_passed boolean,
    summary jsonb,
    created_by uuid NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    started_at timestamp with time zone,
    completed_at timestamp with time zone,
    CONSTRAINT evaluation_executions_agent_snapshot_hash_check CHECK ((length(agent_snapshot_hash) = 64)),
    CONSTRAINT evaluation_executions_agent_version_check CHECK ((agent_version > 0)),
    CONSTRAINT evaluation_executions_candidate_label_check CHECK (((length(candidate_label) >= 1) AND (length(candidate_label) <= 160))),
    CONSTRAINT evaluation_executions_case_count_check CHECK ((case_count > 0)),
    CONSTRAINT evaluation_executions_counts CHECK (((passed_count + failed_count) <= case_count)),
    CONSTRAINT evaluation_executions_failed_count_check CHECK ((failed_count >= 0)),
    CONSTRAINT evaluation_executions_mode_check CHECK ((mode = ANY (ARRAY['deterministic_ci'::text, 'model_dependent'::text]))),
    CONSTRAINT evaluation_executions_passed_count_check CHECK ((passed_count >= 0)),
    CONSTRAINT evaluation_executions_status_check CHECK ((status = ANY (ARRAY['queued'::text, 'running'::text, 'completed'::text, 'failed'::text, 'cancelled'::text]))),
    CONSTRAINT evaluation_executions_terminal CHECK (((status = ANY (ARRAY['completed'::text, 'failed'::text, 'cancelled'::text])) = (completed_at IS NOT NULL)))
);


--
-- Name: evaluation_measurements; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.evaluation_measurements (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    tenant_id uuid NOT NULL,
    execution_id uuid NOT NULL,
    case_run_id uuid NOT NULL,
    metric_name text NOT NULL,
    numeric_value numeric NOT NULL,
    unit text NOT NULL,
    passed boolean NOT NULL,
    details jsonb DEFAULT '{}'::jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT evaluation_measurements_metric_name_check CHECK (((length(metric_name) >= 1) AND (length(metric_name) <= 120))),
    CONSTRAINT evaluation_measurements_unit_check CHECK (((length(unit) >= 1) AND (length(unit) <= 40)))
);


--
-- Name: evaluation_suites; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.evaluation_suites (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    tenant_id uuid NOT NULL,
    logical_id uuid DEFAULT gen_random_uuid() NOT NULL,
    version integer NOT NULL,
    supersedes_id uuid,
    created_by uuid NOT NULL,
    name text NOT NULL,
    description text NOT NULL,
    thresholds jsonb NOT NULL,
    status text DEFAULT 'active'::text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT evaluation_suites_description_check CHECK (((length(description) >= 1) AND (length(description) <= 4000))),
    CONSTRAINT evaluation_suites_name_check CHECK (((length(name) >= 1) AND (length(name) <= 160))),
    CONSTRAINT evaluation_suites_status_check CHECK ((status = ANY (ARRAY['active'::text, 'archived'::text]))),
    CONSTRAINT evaluation_suites_version_check CHECK ((version > 0))
);


--
-- Name: memories; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.memories (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    tenant_id uuid NOT NULL,
    scope text NOT NULL,
    owner_user_id uuid,
    agent_id uuid,
    memory_type text NOT NULL,
    content jsonb NOT NULL,
    content_text text NOT NULL,
    provenance jsonb NOT NULL,
    creation_reason text NOT NULL,
    source_run_id uuid,
    created_by uuid NOT NULL,
    write_source text NOT NULL,
    relevance_metadata jsonb DEFAULT '{}'::jsonb NOT NULL,
    retention_until timestamp with time zone,
    version integer DEFAULT 1 NOT NULL,
    corrected_from_id uuid,
    idempotency_key text,
    deleted_at timestamp with time zone,
    deleted_by uuid,
    deletion_reason text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT memories_content_text_check CHECK (((length(content_text) >= 1) AND (length(content_text) <= 20000))),
    CONSTRAINT memories_creation_reason_check CHECK (((length(creation_reason) >= 1) AND (length(creation_reason) <= 2000))),
    CONSTRAINT memories_deletion_fields CHECK ((((deleted_at IS NULL) AND (deleted_by IS NULL) AND (deletion_reason IS NULL)) OR ((deleted_at IS NOT NULL) AND (deleted_by IS NOT NULL) AND (deletion_reason IS NOT NULL)))),
    CONSTRAINT memories_memory_type_check CHECK ((memory_type = ANY (ARRAY['fact'::text, 'note'::text, 'procedure'::text, 'outcome'::text]))),
    CONSTRAINT memories_scope_check CHECK ((scope = ANY (ARRAY['user'::text, 'agent'::text, 'tenant'::text]))),
    CONSTRAINT memories_scope_owner CHECK ((((scope = 'user'::text) AND (owner_user_id IS NOT NULL) AND (agent_id IS NULL)) OR ((scope = 'agent'::text) AND (owner_user_id IS NULL) AND (agent_id IS NOT NULL)) OR ((scope = 'tenant'::text) AND (owner_user_id IS NULL) AND (agent_id IS NULL)))),
    CONSTRAINT memories_version_check CHECK ((version > 0)),
    CONSTRAINT memories_write_source_check CHECK ((write_source = ANY (ARRAY['user'::text, 'approved_tool'::text, 'system'::text])))
);


--
-- Name: model_attempts; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.model_attempts (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    tenant_id uuid NOT NULL,
    run_id uuid NOT NULL,
    step_id uuid NOT NULL,
    request_id uuid NOT NULL,
    provider_id text NOT NULL,
    model_id text NOT NULL,
    attempt_number integer NOT NULL,
    status public.model_attempt_status NOT NULL,
    started_at timestamp with time zone,
    completed_at timestamp with time zone,
    stop_reason text,
    input_tokens bigint DEFAULT 0 NOT NULL,
    output_tokens bigint DEFAULT 0 NOT NULL,
    cached_tokens bigint DEFAULT 0 NOT NULL,
    cost_microusd bigint DEFAULT 0 NOT NULL,
    normalized_error_code text,
    redacted_metadata jsonb DEFAULT '{}'::jsonb NOT NULL,
    provider_request_id text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT model_attempts_attempt_number_check CHECK ((attempt_number > 0)),
    CONSTRAINT model_attempts_cached_tokens_check CHECK ((cached_tokens >= 0)),
    CONSTRAINT model_attempts_cost_microusd_check CHECK ((cost_microusd >= 0)),
    CONSTRAINT model_attempts_input_tokens_check CHECK ((input_tokens >= 0)),
    CONSTRAINT model_attempts_output_tokens_check CHECK ((output_tokens >= 0))
);


--
-- Name: operational_instances; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.operational_instances (
    instance_id text NOT NULL,
    instance_kind text NOT NULL,
    started_at timestamp with time zone DEFAULT now() NOT NULL,
    last_heartbeat_at timestamp with time zone DEFAULT now() NOT NULL,
    draining boolean DEFAULT false NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb NOT NULL,
    version bigint DEFAULT 1 NOT NULL,
    CONSTRAINT operational_instances_instance_kind_check CHECK ((instance_kind = ANY (ARRAY['worker'::text, 'api'::text]))),
    CONSTRAINT operational_instances_version_check CHECK ((version > 0))
);


--
-- Name: orchestration_waits; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.orchestration_waits (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    tenant_id uuid NOT NULL,
    parent_run_id uuid NOT NULL,
    status text NOT NULL,
    reason text NOT NULL,
    child_snapshot jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    resolved_at timestamp with time zone,
    CONSTRAINT orchestration_waits_reason_check CHECK (((length(reason) >= 1) AND (length(reason) <= 2000))),
    CONSTRAINT orchestration_waits_resolution CHECK (((status = 'waiting'::text) = (resolved_at IS NULL))),
    CONSTRAINT orchestration_waits_status_check CHECK ((status = ANY (ARRAY['waiting'::text, 'resolved'::text, 'cancelled'::text])))
);


--
-- Name: revoked_tokens; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.revoked_tokens (
    issuer text NOT NULL,
    token_id text NOT NULL,
    expires_at timestamp with time zone NOT NULL,
    revoked_at timestamp with time zone DEFAULT now() NOT NULL,
    reason text
);


--
-- Name: run_delegations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.run_delegations (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    tenant_id uuid NOT NULL,
    parent_run_id uuid NOT NULL,
    child_run_id uuid NOT NULL,
    parent_step_id uuid NOT NULL,
    target_agent_id uuid NOT NULL,
    idempotency_key text NOT NULL,
    role_name text NOT NULL,
    required boolean DEFAULT true NOT NULL,
    context_scope text NOT NULL,
    shared_context jsonb,
    token_budget bigint NOT NULL,
    cost_budget_microusd bigint NOT NULL,
    created_by uuid NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT run_delegations_context_scope_check CHECK ((context_scope = ANY (ARRAY['private'::text, 'shared'::text]))),
    CONSTRAINT run_delegations_cost_budget_microusd_check CHECK ((cost_budget_microusd >= 0)),
    CONSTRAINT run_delegations_idempotency_key_check CHECK (((length(idempotency_key) >= 1) AND (length(idempotency_key) <= 500))),
    CONSTRAINT run_delegations_role_name_check CHECK (((length(role_name) >= 1) AND (length(role_name) <= 120))),
    CONSTRAINT run_delegations_shared_context CHECK (((context_scope = 'shared'::text) = (shared_context IS NOT NULL))),
    CONSTRAINT run_delegations_token_budget_check CHECK ((token_budget > 0))
);


--
-- Name: run_plans; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.run_plans (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    tenant_id uuid NOT NULL,
    run_id uuid NOT NULL,
    version integer NOT NULL,
    status text NOT NULL,
    objective text NOT NULL,
    plan jsonb NOT NULL,
    created_by_step_id uuid,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    superseded_at timestamp with time zone,
    CONSTRAINT run_plans_status_check CHECK ((status = ANY (ARRAY['active'::text, 'superseded'::text]))),
    CONSTRAINT run_plans_version_check CHECK ((version > 0))
);


--
-- Name: runs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.runs (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    tenant_id uuid NOT NULL,
    agent_id uuid NOT NULL,
    created_by uuid NOT NULL,
    goal text NOT NULL,
    status public.run_status DEFAULT 'queued'::public.run_status NOT NULL,
    current_step integer DEFAULT 0 NOT NULL,
    version integer DEFAULT 1 NOT NULL,
    lease_owner text,
    lease_expires_at timestamp with time zone,
    cancellation_requested_at timestamp with time zone,
    input_tokens bigint DEFAULT 0 NOT NULL,
    output_tokens bigint DEFAULT 0 NOT NULL,
    cost_microusd bigint DEFAULT 0 NOT NULL,
    final_output jsonb,
    error_details jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    completed_at timestamp with time zone,
    parent_run_id uuid,
    root_run_id uuid NOT NULL,
    delegation_depth integer DEFAULT 0 NOT NULL,
    delegation_role text,
    context_scope text DEFAULT 'private'::text NOT NULL,
    token_budget_limit bigint NOT NULL,
    cost_budget_limit_microusd bigint NOT NULL,
    reserved_child_tokens bigint DEFAULT 0 NOT NULL,
    reserved_child_cost_microusd bigint DEFAULT 0 NOT NULL,
    agent_version integer NOT NULL,
    agent_configuration_snapshot jsonb NOT NULL,
    CONSTRAINT cancelled_cannot_complete CHECK (((status <> 'completed'::public.run_status) OR (cancellation_requested_at IS NULL))),
    CONSTRAINT final_output_only_completed CHECK (((final_output IS NULL) OR (status = 'completed'::public.run_status))),
    CONSTRAINT lease_fields_together CHECK (((lease_owner IS NULL) = (lease_expires_at IS NULL))),
    CONSTRAINT lease_only_active CHECK (((status = ANY (ARRAY['claimed'::public.run_status, 'running'::public.run_status, 'cancelling'::public.run_status])) OR ((lease_owner IS NULL) AND (lease_expires_at IS NULL)))),
    CONSTRAINT runs_agent_version_positive CHECK ((agent_version > 0)),
    CONSTRAINT runs_budget_limits CHECK (((token_budget_limit >= 0) AND (cost_budget_limit_microusd >= 0))),
    CONSTRAINT runs_context_scope_check CHECK ((context_scope = ANY (ARRAY['private'::text, 'shared'::text]))),
    CONSTRAINT runs_cost_microusd_check CHECK ((cost_microusd >= 0)),
    CONSTRAINT runs_current_step_check CHECK ((current_step >= 0)),
    CONSTRAINT runs_delegation_depth_check CHECK (((delegation_depth >= 0) AND (delegation_depth <= 16))),
    CONSTRAINT runs_delegation_role_check CHECK (((delegation_role IS NULL) OR ((length(delegation_role) >= 1) AND (length(delegation_role) <= 120)))),
    CONSTRAINT runs_goal_check CHECK ((length(goal) > 0)),
    CONSTRAINT runs_hierarchy_shape CHECK ((((parent_run_id IS NULL) AND (root_run_id = id) AND (delegation_depth = 0) AND (delegation_role IS NULL) AND (context_scope = 'private'::text)) OR ((parent_run_id IS NOT NULL) AND (delegation_depth > 0) AND (delegation_role IS NOT NULL)))),
    CONSTRAINT runs_input_tokens_check CHECK ((input_tokens >= 0)),
    CONSTRAINT runs_output_tokens_check CHECK ((output_tokens >= 0)),
    CONSTRAINT runs_reserved_child_cost_microusd_check CHECK ((reserved_child_cost_microusd >= 0)),
    CONSTRAINT runs_reserved_child_tokens_check CHECK ((reserved_child_tokens >= 0)),
    CONSTRAINT runs_version_check CHECK ((version > 0))
);


--
-- Name: schema_migrations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.schema_migrations (
    version text NOT NULL,
    applied_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: skills; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.skills (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    tenant_id uuid NOT NULL,
    logical_id uuid DEFAULT gen_random_uuid() NOT NULL,
    version integer NOT NULL,
    name text NOT NULL,
    instructions text NOT NULL,
    allowed_tools jsonb DEFAULT '[]'::jsonb NOT NULL,
    provenance jsonb NOT NULL,
    content_hash text NOT NULL,
    created_by uuid NOT NULL,
    supersedes_id uuid,
    is_current boolean DEFAULT false NOT NULL,
    revoked_at timestamp with time zone,
    revoked_by uuid,
    revocation_reason text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT skills_content_hash_check CHECK ((content_hash ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT skills_instructions_check CHECK (((length(instructions) >= 1) AND (length(instructions) <= 20000))),
    CONSTRAINT skills_name_check CHECK (((length(name) >= 1) AND (length(name) <= 200))),
    CONSTRAINT skills_revocation_fields CHECK ((((revoked_at IS NULL) AND (revoked_by IS NULL) AND (revocation_reason IS NULL)) OR ((revoked_at IS NOT NULL) AND (revoked_by IS NOT NULL) AND (revocation_reason IS NOT NULL)))),
    CONSTRAINT skills_version_check CHECK ((version > 0))
);


--
-- Name: steps; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.steps (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    tenant_id uuid NOT NULL,
    run_id uuid NOT NULL,
    sequence integer NOT NULL,
    kind text NOT NULL,
    status public.step_status NOT NULL,
    idempotency_key text NOT NULL,
    attempt_count integer DEFAULT 0 NOT NULL,
    input jsonb,
    output jsonb,
    error_details jsonb,
    started_at timestamp with time zone,
    finished_at timestamp with time zone,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT steps_attempt_count_check CHECK ((attempt_count >= 0)),
    CONSTRAINT steps_sequence_check CHECK ((sequence > 0))
);


--
-- Name: structured_notes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.structured_notes (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    tenant_id uuid NOT NULL,
    created_by uuid NOT NULL,
    title text NOT NULL,
    body text NOT NULL,
    tags jsonb DEFAULT '[]'::jsonb NOT NULL,
    idempotency_key text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT structured_notes_body_check CHECK (((length(body) >= 1) AND (length(body) <= 10000))),
    CONSTRAINT structured_notes_title_check CHECK (((length(title) >= 1) AND (length(title) <= 200)))
);


--
-- Name: tenant_memberships; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.tenant_memberships (
    tenant_id uuid NOT NULL,
    identity_id uuid NOT NULL,
    identity_type text DEFAULT 'user'::text NOT NULL,
    roles jsonb DEFAULT '[]'::jsonb NOT NULL,
    status text DEFAULT 'active'::text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT tenant_memberships_identity_type_check CHECK ((identity_type = ANY (ARRAY['user'::text, 'service'::text]))),
    CONSTRAINT tenant_memberships_roles_check CHECK ((jsonb_typeof(roles) = 'array'::text)),
    CONSTRAINT tenant_memberships_status_check CHECK ((status = ANY (ARRAY['active'::text, 'revoked'::text])))
);


--
-- Name: tool_execution_attempts; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.tool_execution_attempts (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    tenant_id uuid NOT NULL,
    run_id uuid NOT NULL,
    tool_execution_id uuid NOT NULL,
    attempt_number integer NOT NULL,
    status public.step_status NOT NULL,
    started_at timestamp with time zone DEFAULT now() NOT NULL,
    completed_at timestamp with time zone,
    error_code text,
    error_details jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT tool_execution_attempts_attempt_number_check CHECK ((attempt_number > 0))
);


--
-- Name: tool_executions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.tool_executions (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    tenant_id uuid NOT NULL,
    run_id uuid NOT NULL,
    step_id uuid NOT NULL,
    tool_name text NOT NULL,
    idempotency_key text NOT NULL,
    status public.step_status NOT NULL,
    validated_arguments jsonb NOT NULL,
    output jsonb,
    error_details jsonb,
    duration_ms integer,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    operation_hash text NOT NULL,
    attempt_count integer DEFAULT 0 NOT NULL,
    reconciliation_status text DEFAULT 'not_required'::text NOT NULL,
    reconciliation_owner text,
    reconciliation_expires_at timestamp with time zone,
    reconciled_at timestamp with time zone,
    retry_safety text DEFAULT 'non_retryable'::text NOT NULL,
    CONSTRAINT tool_execution_operation_hash_format CHECK ((operation_hash ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT tool_execution_reconciliation_lease CHECK (((reconciliation_owner IS NULL) = (reconciliation_expires_at IS NULL))),
    CONSTRAINT tool_execution_reconciliation_status CHECK ((reconciliation_status = ANY (ARRAY['not_required'::text, 'pending'::text, 'running'::text, 'resolved'::text, 'manual'::text]))),
    CONSTRAINT tool_execution_retry_safety CHECK ((retry_safety = ANY (ARRAY['pure'::text, 'externally_idempotent'::text, 'reconcilable'::text, 'non_retryable'::text]))),
    CONSTRAINT tool_executions_attempt_count_check CHECK ((attempt_count >= 0)),
    CONSTRAINT tool_executions_duration_ms_check CHECK ((duration_ms >= 0))
);


--
-- Name: usage_records; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.usage_records (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    tenant_id uuid NOT NULL,
    run_id uuid NOT NULL,
    step_id uuid,
    provider text NOT NULL,
    model text NOT NULL,
    input_tokens bigint NOT NULL,
    output_tokens bigint NOT NULL,
    cost_microusd bigint NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT usage_records_cost_microusd_check CHECK ((cost_microusd >= 0)),
    CONSTRAINT usage_records_input_tokens_check CHECK ((input_tokens >= 0)),
    CONSTRAINT usage_records_output_tokens_check CHECK ((output_tokens >= 0))
);


--
-- Name: worker_leases; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.worker_leases (
    run_id uuid NOT NULL,
    tenant_id uuid NOT NULL,
    worker_id text NOT NULL,
    acquired_at timestamp with time zone NOT NULL,
    heartbeat_at timestamp with time zone NOT NULL,
    expires_at timestamp with time zone NOT NULL,
    lease_generation bigint NOT NULL,
    CONSTRAINT worker_leases_lease_generation_check CHECK ((lease_generation > 0))
);


--
-- Name: agents agents_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.agents
    ADD CONSTRAINT agents_pkey PRIMARY KEY (id);


--
-- Name: agents agents_tenant_id_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.agents
    ADD CONSTRAINT agents_tenant_id_id_key UNIQUE (tenant_id, id);


--
-- Name: agents agents_tenant_id_name_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.agents
    ADD CONSTRAINT agents_tenant_id_name_key UNIQUE (tenant_id, name);


--
-- Name: approvals approvals_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.approvals
    ADD CONSTRAINT approvals_pkey PRIMARY KEY (id);


--
-- Name: approvals approvals_step_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.approvals
    ADD CONSTRAINT approvals_step_id_key UNIQUE (step_id);


--
-- Name: approvals approvals_tenant_id_unique; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.approvals
    ADD CONSTRAINT approvals_tenant_id_unique UNIQUE (tenant_id, id);


--
-- Name: approvals approvals_tenant_idempotency_unique; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.approvals
    ADD CONSTRAINT approvals_tenant_idempotency_unique UNIQUE (tenant_id, idempotency_key);


--
-- Name: audit_events audit_events_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.audit_events
    ADD CONSTRAINT audit_events_pkey PRIMARY KEY (id);


--
-- Name: checkpoints checkpoints_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.checkpoints
    ADD CONSTRAINT checkpoints_pkey PRIMARY KEY (id);


--
-- Name: checkpoints checkpoints_run_version_unique; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.checkpoints
    ADD CONSTRAINT checkpoints_run_version_unique UNIQUE (run_id, version);


--
-- Name: connector_invocations connector_invocations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.connector_invocations
    ADD CONSTRAINT connector_invocations_pkey PRIMARY KEY (id);


--
-- Name: connector_invocations connector_invocations_tenant_id_connector_id_idempotency_ke_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.connector_invocations
    ADD CONSTRAINT connector_invocations_tenant_id_connector_id_idempotency_ke_key UNIQUE (tenant_id, connector_id, idempotency_key);


--
-- Name: connector_invocations connector_invocations_tenant_id_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.connector_invocations
    ADD CONSTRAINT connector_invocations_tenant_id_id_key UNIQUE (tenant_id, id);


--
-- Name: connector_rate_limits connector_rate_limits_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.connector_rate_limits
    ADD CONSTRAINT connector_rate_limits_pkey PRIMARY KEY (tenant_id, connector_id, window_start);


--
-- Name: connector_tools connector_tools_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.connector_tools
    ADD CONSTRAINT connector_tools_pkey PRIMARY KEY (id);


--
-- Name: connector_tools connector_tools_tenant_id_connector_id_name_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.connector_tools
    ADD CONSTRAINT connector_tools_tenant_id_connector_id_name_key UNIQUE (tenant_id, connector_id, name);


--
-- Name: connector_tools connector_tools_tenant_id_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.connector_tools
    ADD CONSTRAINT connector_tools_tenant_id_id_key UNIQUE (tenant_id, id);


--
-- Name: connectors connectors_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.connectors
    ADD CONSTRAINT connectors_pkey PRIMARY KEY (id);


--
-- Name: connectors connectors_tenant_id_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.connectors
    ADD CONSTRAINT connectors_tenant_id_id_key UNIQUE (tenant_id, id);


--
-- Name: connectors connectors_tenant_id_name_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.connectors
    ADD CONSTRAINT connectors_tenant_id_name_key UNIQUE (tenant_id, name);


--
-- Name: context_builds context_builds_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.context_builds
    ADD CONSTRAINT context_builds_pkey PRIMARY KEY (id);


--
-- Name: context_builds context_builds_tenant_id_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.context_builds
    ADD CONSTRAINT context_builds_tenant_id_id_key UNIQUE (tenant_id, id);


--
-- Name: context_summaries context_summaries_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.context_summaries
    ADD CONSTRAINT context_summaries_pkey PRIMARY KEY (id);


--
-- Name: context_summaries context_summaries_run_id_through_sequence_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.context_summaries
    ADD CONSTRAINT context_summaries_run_id_through_sequence_key UNIQUE (run_id, through_sequence);


--
-- Name: context_summaries context_summaries_tenant_id_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.context_summaries
    ADD CONSTRAINT context_summaries_tenant_id_id_key UNIQUE (tenant_id, id);


--
-- Name: demo_products demo_products_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.demo_products
    ADD CONSTRAINT demo_products_pkey PRIMARY KEY (id);


--
-- Name: document_chunks document_chunks_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_chunks
    ADD CONSTRAINT document_chunks_pkey PRIMARY KEY (id);


--
-- Name: document_chunks document_chunks_tenant_id_document_id_chunk_index_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_chunks
    ADD CONSTRAINT document_chunks_tenant_id_document_id_chunk_index_key UNIQUE (tenant_id, document_id, chunk_index);


--
-- Name: document_chunks document_chunks_tenant_id_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_chunks
    ADD CONSTRAINT document_chunks_tenant_id_id_key UNIQUE (tenant_id, id);


--
-- Name: documents documents_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.documents
    ADD CONSTRAINT documents_pkey PRIMARY KEY (id);


--
-- Name: documents documents_tenant_id_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.documents
    ADD CONSTRAINT documents_tenant_id_id_key UNIQUE (tenant_id, id);


--
-- Name: documents documents_tenant_id_logical_id_version_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.documents
    ADD CONSTRAINT documents_tenant_id_logical_id_version_key UNIQUE (tenant_id, logical_id, version);


--
-- Name: evaluation_case_runs evaluation_case_runs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_case_runs
    ADD CONSTRAINT evaluation_case_runs_pkey PRIMARY KEY (id);


--
-- Name: evaluation_case_runs evaluation_case_runs_tenant_id_execution_id_case_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_case_runs
    ADD CONSTRAINT evaluation_case_runs_tenant_id_execution_id_case_id_key UNIQUE (tenant_id, execution_id, case_id);


--
-- Name: evaluation_case_runs evaluation_case_runs_tenant_id_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_case_runs
    ADD CONSTRAINT evaluation_case_runs_tenant_id_id_key UNIQUE (tenant_id, id);


--
-- Name: evaluation_case_runs evaluation_case_runs_tenant_id_run_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_case_runs
    ADD CONSTRAINT evaluation_case_runs_tenant_id_run_id_key UNIQUE (tenant_id, run_id);


--
-- Name: evaluation_cases evaluation_cases_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_cases
    ADD CONSTRAINT evaluation_cases_pkey PRIMARY KEY (id);


--
-- Name: evaluation_cases evaluation_cases_tenant_id_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_cases
    ADD CONSTRAINT evaluation_cases_tenant_id_id_key UNIQUE (tenant_id, id);


--
-- Name: evaluation_cases evaluation_cases_tenant_id_suite_id_ordinal_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_cases
    ADD CONSTRAINT evaluation_cases_tenant_id_suite_id_ordinal_key UNIQUE (tenant_id, suite_id, ordinal);


--
-- Name: evaluation_executions evaluation_executions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_executions
    ADD CONSTRAINT evaluation_executions_pkey PRIMARY KEY (id);


--
-- Name: evaluation_executions evaluation_executions_tenant_id_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_executions
    ADD CONSTRAINT evaluation_executions_tenant_id_id_key UNIQUE (tenant_id, id);


--
-- Name: evaluation_measurements evaluation_measurements_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_measurements
    ADD CONSTRAINT evaluation_measurements_pkey PRIMARY KEY (id);


--
-- Name: evaluation_measurements evaluation_measurements_tenant_id_case_run_id_metric_name_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_measurements
    ADD CONSTRAINT evaluation_measurements_tenant_id_case_run_id_metric_name_key UNIQUE (tenant_id, case_run_id, metric_name);


--
-- Name: evaluation_measurements evaluation_measurements_tenant_id_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_measurements
    ADD CONSTRAINT evaluation_measurements_tenant_id_id_key UNIQUE (tenant_id, id);


--
-- Name: evaluation_suites evaluation_suites_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_suites
    ADD CONSTRAINT evaluation_suites_pkey PRIMARY KEY (id);


--
-- Name: evaluation_suites evaluation_suites_tenant_id_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_suites
    ADD CONSTRAINT evaluation_suites_tenant_id_id_key UNIQUE (tenant_id, id);


--
-- Name: evaluation_suites evaluation_suites_tenant_id_logical_id_version_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_suites
    ADD CONSTRAINT evaluation_suites_tenant_id_logical_id_version_key UNIQUE (tenant_id, logical_id, version);


--
-- Name: memories memories_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.memories
    ADD CONSTRAINT memories_pkey PRIMARY KEY (id);


--
-- Name: memories memories_tenant_id_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.memories
    ADD CONSTRAINT memories_tenant_id_id_key UNIQUE (tenant_id, id);


--
-- Name: model_attempts model_attempts_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_attempts
    ADD CONSTRAINT model_attempts_pkey PRIMARY KEY (id);


--
-- Name: model_attempts model_attempts_run_id_request_id_attempt_number_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_attempts
    ADD CONSTRAINT model_attempts_run_id_request_id_attempt_number_key UNIQUE (run_id, request_id, attempt_number);


--
-- Name: model_attempts model_attempts_tenant_id_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_attempts
    ADD CONSTRAINT model_attempts_tenant_id_id_key UNIQUE (tenant_id, id);


--
-- Name: operational_instances operational_instances_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.operational_instances
    ADD CONSTRAINT operational_instances_pkey PRIMARY KEY (instance_id);


--
-- Name: orchestration_waits orchestration_waits_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.orchestration_waits
    ADD CONSTRAINT orchestration_waits_pkey PRIMARY KEY (id);


--
-- Name: orchestration_waits orchestration_waits_tenant_id_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.orchestration_waits
    ADD CONSTRAINT orchestration_waits_tenant_id_id_key UNIQUE (tenant_id, id);


--
-- Name: revoked_tokens revoked_tokens_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.revoked_tokens
    ADD CONSTRAINT revoked_tokens_pkey PRIMARY KEY (issuer, token_id);


--
-- Name: run_delegations run_delegations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.run_delegations
    ADD CONSTRAINT run_delegations_pkey PRIMARY KEY (id);


--
-- Name: run_delegations run_delegations_tenant_id_child_run_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.run_delegations
    ADD CONSTRAINT run_delegations_tenant_id_child_run_id_key UNIQUE (tenant_id, child_run_id);


--
-- Name: run_delegations run_delegations_tenant_id_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.run_delegations
    ADD CONSTRAINT run_delegations_tenant_id_id_key UNIQUE (tenant_id, id);


--
-- Name: run_delegations run_delegations_tenant_id_parent_run_id_idempotency_key_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.run_delegations
    ADD CONSTRAINT run_delegations_tenant_id_parent_run_id_idempotency_key_key UNIQUE (tenant_id, parent_run_id, idempotency_key);


--
-- Name: run_plans run_plans_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.run_plans
    ADD CONSTRAINT run_plans_pkey PRIMARY KEY (id);


--
-- Name: run_plans run_plans_run_id_version_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.run_plans
    ADD CONSTRAINT run_plans_run_id_version_key UNIQUE (run_id, version);


--
-- Name: run_plans run_plans_tenant_id_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.run_plans
    ADD CONSTRAINT run_plans_tenant_id_id_key UNIQUE (tenant_id, id);


--
-- Name: runs runs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.runs
    ADD CONSTRAINT runs_pkey PRIMARY KEY (id);


--
-- Name: runs runs_tenant_id_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.runs
    ADD CONSTRAINT runs_tenant_id_id_key UNIQUE (tenant_id, id);


--
-- Name: schema_migrations schema_migrations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.schema_migrations
    ADD CONSTRAINT schema_migrations_pkey PRIMARY KEY (version);


--
-- Name: skills skills_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.skills
    ADD CONSTRAINT skills_pkey PRIMARY KEY (id);


--
-- Name: skills skills_tenant_id_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.skills
    ADD CONSTRAINT skills_tenant_id_id_key UNIQUE (tenant_id, id);


--
-- Name: skills skills_tenant_id_logical_id_version_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.skills
    ADD CONSTRAINT skills_tenant_id_logical_id_version_key UNIQUE (tenant_id, logical_id, version);


--
-- Name: steps steps_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.steps
    ADD CONSTRAINT steps_pkey PRIMARY KEY (id);


--
-- Name: steps steps_run_id_idempotency_key_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.steps
    ADD CONSTRAINT steps_run_id_idempotency_key_key UNIQUE (run_id, idempotency_key);


--
-- Name: steps steps_run_id_sequence_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.steps
    ADD CONSTRAINT steps_run_id_sequence_key UNIQUE (run_id, sequence);


--
-- Name: steps steps_tenant_id_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.steps
    ADD CONSTRAINT steps_tenant_id_id_key UNIQUE (tenant_id, id);


--
-- Name: structured_notes structured_notes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.structured_notes
    ADD CONSTRAINT structured_notes_pkey PRIMARY KEY (id);


--
-- Name: structured_notes structured_notes_tenant_id_idempotency_key_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.structured_notes
    ADD CONSTRAINT structured_notes_tenant_id_idempotency_key_key UNIQUE (tenant_id, idempotency_key);


--
-- Name: tenant_memberships tenant_memberships_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tenant_memberships
    ADD CONSTRAINT tenant_memberships_pkey PRIMARY KEY (tenant_id, identity_id);


--
-- Name: tool_execution_attempts tool_execution_attempts_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tool_execution_attempts
    ADD CONSTRAINT tool_execution_attempts_pkey PRIMARY KEY (id);


--
-- Name: tool_execution_attempts tool_execution_attempts_tenant_id_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tool_execution_attempts
    ADD CONSTRAINT tool_execution_attempts_tenant_id_id_key UNIQUE (tenant_id, id);


--
-- Name: tool_execution_attempts tool_execution_attempts_tool_execution_id_attempt_number_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tool_execution_attempts
    ADD CONSTRAINT tool_execution_attempts_tool_execution_id_attempt_number_key UNIQUE (tool_execution_id, attempt_number);


--
-- Name: tool_executions tool_executions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tool_executions
    ADD CONSTRAINT tool_executions_pkey PRIMARY KEY (id);


--
-- Name: tool_executions tool_executions_tenant_id_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tool_executions
    ADD CONSTRAINT tool_executions_tenant_id_id_key UNIQUE (tenant_id, id);


--
-- Name: tool_executions tool_executions_tenant_id_idempotency_key_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tool_executions
    ADD CONSTRAINT tool_executions_tenant_id_idempotency_key_key UNIQUE (tenant_id, idempotency_key);


--
-- Name: usage_records usage_records_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.usage_records
    ADD CONSTRAINT usage_records_pkey PRIMARY KEY (id);


--
-- Name: worker_leases worker_leases_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.worker_leases
    ADD CONSTRAINT worker_leases_pkey PRIMARY KEY (run_id);


--
-- Name: approvals_pending_run_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX approvals_pending_run_idx ON public.approvals USING btree (tenant_id, run_id, created_at) WHERE (decision = 'pending'::public.approval_decision);


--
-- Name: audit_run_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX audit_run_idx ON public.audit_events USING btree (tenant_id, run_id, created_at);


--
-- Name: connector_invocations_run_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX connector_invocations_run_idx ON public.connector_invocations USING btree (tenant_id, run_id, created_at);


--
-- Name: connector_tools_lookup_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX connector_tools_lookup_idx ON public.connector_tools USING btree (tenant_id, connector_id, name) WHERE enabled;


--
-- Name: document_chunks_search_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX document_chunks_search_idx ON public.document_chunks USING btree (tenant_id, embedding_provider, embedding_model, embedding_dimension) WHERE (deleted_at IS NULL);


--
-- Name: documents_current_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX documents_current_idx ON public.documents USING btree (tenant_id, logical_id) WHERE (is_current AND (status <> 'deleted'::text));


--
-- Name: documents_owner_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX documents_owner_idx ON public.documents USING btree (tenant_id, owner_user_id, created_at DESC);


--
-- Name: evaluation_case_runs_pending_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX evaluation_case_runs_pending_idx ON public.evaluation_case_runs USING btree (status, created_at, id) WHERE (status = ANY (ARRAY['pending'::text, 'scoring'::text]));


--
-- Name: evaluation_executions_history_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX evaluation_executions_history_idx ON public.evaluation_executions USING btree (tenant_id, suite_id, created_at DESC, id);


--
-- Name: evaluation_measurements_metric_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX evaluation_measurements_metric_idx ON public.evaluation_measurements USING btree (tenant_id, metric_name, created_at);


--
-- Name: evaluation_suites_history_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX evaluation_suites_history_idx ON public.evaluation_suites USING btree (tenant_id, logical_id, version DESC);


--
-- Name: memories_active_scope_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX memories_active_scope_idx ON public.memories USING btree (tenant_id, scope, agent_id, owner_user_id, created_at DESC) WHERE (deleted_at IS NULL);


--
-- Name: memories_idempotency_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX memories_idempotency_idx ON public.memories USING btree (tenant_id, idempotency_key) WHERE (idempotency_key IS NOT NULL);


--
-- Name: memories_retention_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX memories_retention_idx ON public.memories USING btree (retention_until) WHERE ((retention_until IS NOT NULL) AND (deleted_at IS NULL));


--
-- Name: model_attempts_operational_failure_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX model_attempts_operational_failure_idx ON public.model_attempts USING btree (tenant_id, normalized_error_code, completed_at) WHERE (status = 'failed'::public.model_attempt_status);


--
-- Name: model_attempts_run_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX model_attempts_run_idx ON public.model_attempts USING btree (tenant_id, run_id, created_at);


--
-- Name: operational_instances_health_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX operational_instances_health_idx ON public.operational_instances USING btree (instance_kind, last_heartbeat_at DESC);


--
-- Name: orchestration_waits_active_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX orchestration_waits_active_idx ON public.orchestration_waits USING btree (tenant_id, parent_run_id) WHERE (status = 'waiting'::text);


--
-- Name: orchestration_waits_scan_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX orchestration_waits_scan_idx ON public.orchestration_waits USING btree (created_at, id) WHERE (status = 'waiting'::text);


--
-- Name: revoked_tokens_expiry_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX revoked_tokens_expiry_idx ON public.revoked_tokens USING btree (expires_at);


--
-- Name: run_delegations_parent_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX run_delegations_parent_idx ON public.run_delegations USING btree (tenant_id, parent_run_id, created_at, id);


--
-- Name: run_plans_one_active_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX run_plans_one_active_idx ON public.run_plans USING btree (run_id) WHERE (status = 'active'::text);


--
-- Name: runs_claim_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX runs_claim_idx ON public.runs USING btree (created_at, id) WHERE ((status = 'queued'::public.run_status) AND (cancellation_requested_at IS NULL));


--
-- Name: runs_expired_lease_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX runs_expired_lease_idx ON public.runs USING btree (lease_expires_at) WHERE (lease_expires_at IS NOT NULL);


--
-- Name: runs_operational_queue_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX runs_operational_queue_idx ON public.runs USING btree (tenant_id, status, created_at);


--
-- Name: runs_parent_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX runs_parent_idx ON public.runs USING btree (tenant_id, parent_run_id, created_at, id) WHERE (parent_run_id IS NOT NULL);


--
-- Name: runs_root_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX runs_root_idx ON public.runs USING btree (tenant_id, root_run_id, delegation_depth, created_at, id);


--
-- Name: skills_current_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX skills_current_idx ON public.skills USING btree (tenant_id, logical_id) WHERE (is_current AND (revoked_at IS NULL));


--
-- Name: skills_current_name_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX skills_current_name_idx ON public.skills USING btree (tenant_id, lower(name)) WHERE (is_current AND (revoked_at IS NULL));


--
-- Name: tool_executions_reconcile_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX tool_executions_reconcile_idx ON public.tool_executions USING btree (updated_at, id) WHERE ((status = 'unknown'::public.step_status) AND (reconciliation_status = ANY (ARRAY['pending'::text, 'running'::text])));


--
-- Name: worker_leases_operational_health_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX worker_leases_operational_health_idx ON public.worker_leases USING btree (tenant_id, expires_at);


--
-- Name: approvals approvals_frozen_action_guard; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER approvals_frozen_action_guard BEFORE UPDATE ON public.approvals FOR EACH ROW EXECUTE FUNCTION public.protect_frozen_approval_action();


--
-- Name: evaluation_case_runs evaluation_case_run_identity_immutable; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER evaluation_case_run_identity_immutable BEFORE UPDATE ON public.evaluation_case_runs FOR EACH ROW EXECUTE FUNCTION public.protect_evaluation_case_run_identity();


--
-- Name: evaluation_cases evaluation_cases_append_only; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER evaluation_cases_append_only BEFORE UPDATE ON public.evaluation_cases FOR EACH ROW EXECUTE FUNCTION public.reject_evaluation_evidence_update();


--
-- Name: evaluation_executions evaluation_execution_identity_immutable; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER evaluation_execution_identity_immutable BEFORE UPDATE ON public.evaluation_executions FOR EACH ROW EXECUTE FUNCTION public.protect_evaluation_execution_identity();


--
-- Name: evaluation_measurements evaluation_measurements_append_only; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER evaluation_measurements_append_only BEFORE UPDATE ON public.evaluation_measurements FOR EACH ROW EXECUTE FUNCTION public.reject_evaluation_evidence_update();


--
-- Name: evaluation_suites evaluation_suite_version_immutable; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER evaluation_suite_version_immutable BEFORE UPDATE ON public.evaluation_suites FOR EACH ROW EXECUTE FUNCTION public.protect_evaluation_suite_version();


--
-- Name: runs runs_orchestrated_completion_guard; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER runs_orchestrated_completion_guard BEFORE UPDATE OF status ON public.runs FOR EACH ROW EXECUTE FUNCTION public.enforce_orchestrated_completion();


--
-- Name: runs runs_transition_guard; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER runs_transition_guard BEFORE UPDATE OF status ON public.runs FOR EACH ROW EXECUTE FUNCTION public.enforce_run_transition();


--
-- Name: approvals approvals_run_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.approvals
    ADD CONSTRAINT approvals_run_fk FOREIGN KEY (tenant_id, run_id) REFERENCES public.runs(tenant_id, id);


--
-- Name: approvals approvals_step_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.approvals
    ADD CONSTRAINT approvals_step_fk FOREIGN KEY (tenant_id, step_id) REFERENCES public.steps(tenant_id, id);


--
-- Name: checkpoints checkpoints_run_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.checkpoints
    ADD CONSTRAINT checkpoints_run_fk FOREIGN KEY (tenant_id, run_id) REFERENCES public.runs(tenant_id, id);


--
-- Name: checkpoints checkpoints_step_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.checkpoints
    ADD CONSTRAINT checkpoints_step_fk FOREIGN KEY (tenant_id, step_id) REFERENCES public.steps(tenant_id, id);


--
-- Name: connector_invocations connector_invocations_connector_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.connector_invocations
    ADD CONSTRAINT connector_invocations_connector_fk FOREIGN KEY (tenant_id, connector_id) REFERENCES public.connectors(tenant_id, id);


--
-- Name: connector_invocations connector_invocations_run_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.connector_invocations
    ADD CONSTRAINT connector_invocations_run_fk FOREIGN KEY (tenant_id, run_id) REFERENCES public.runs(tenant_id, id);


--
-- Name: connector_invocations connector_invocations_tool_execution_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.connector_invocations
    ADD CONSTRAINT connector_invocations_tool_execution_fk FOREIGN KEY (tenant_id, tool_execution_id) REFERENCES public.tool_executions(tenant_id, id);


--
-- Name: connector_rate_limits connector_rate_limits_connector_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.connector_rate_limits
    ADD CONSTRAINT connector_rate_limits_connector_fk FOREIGN KEY (tenant_id, connector_id) REFERENCES public.connectors(tenant_id, id) ON DELETE CASCADE;


--
-- Name: connector_tools connector_tools_connector_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.connector_tools
    ADD CONSTRAINT connector_tools_connector_fk FOREIGN KEY (tenant_id, connector_id) REFERENCES public.connectors(tenant_id, id) ON DELETE CASCADE;


--
-- Name: context_builds context_builds_checkpoint_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.context_builds
    ADD CONSTRAINT context_builds_checkpoint_fk FOREIGN KEY (checkpoint_id) REFERENCES public.checkpoints(id);


--
-- Name: context_builds context_builds_plan_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.context_builds
    ADD CONSTRAINT context_builds_plan_fk FOREIGN KEY (tenant_id, plan_id) REFERENCES public.run_plans(tenant_id, id);


--
-- Name: context_builds context_builds_run_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.context_builds
    ADD CONSTRAINT context_builds_run_fk FOREIGN KEY (tenant_id, run_id) REFERENCES public.runs(tenant_id, id);


--
-- Name: context_builds context_builds_summary_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.context_builds
    ADD CONSTRAINT context_builds_summary_fk FOREIGN KEY (tenant_id, summary_id) REFERENCES public.context_summaries(tenant_id, id);


--
-- Name: context_summaries context_summaries_run_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.context_summaries
    ADD CONSTRAINT context_summaries_run_fk FOREIGN KEY (tenant_id, run_id) REFERENCES public.runs(tenant_id, id);


--
-- Name: document_chunks document_chunks_document_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_chunks
    ADD CONSTRAINT document_chunks_document_fk FOREIGN KEY (tenant_id, document_id) REFERENCES public.documents(tenant_id, id) ON DELETE CASCADE;


--
-- Name: documents documents_supersedes_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.documents
    ADD CONSTRAINT documents_supersedes_fk FOREIGN KEY (tenant_id, supersedes_id) REFERENCES public.documents(tenant_id, id);


--
-- Name: evaluation_case_runs evaluation_case_runs_case_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_case_runs
    ADD CONSTRAINT evaluation_case_runs_case_fk FOREIGN KEY (tenant_id, case_id) REFERENCES public.evaluation_cases(tenant_id, id);


--
-- Name: evaluation_case_runs evaluation_case_runs_execution_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_case_runs
    ADD CONSTRAINT evaluation_case_runs_execution_fk FOREIGN KEY (tenant_id, execution_id) REFERENCES public.evaluation_executions(tenant_id, id);


--
-- Name: evaluation_case_runs evaluation_case_runs_run_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_case_runs
    ADD CONSTRAINT evaluation_case_runs_run_fk FOREIGN KEY (tenant_id, run_id) REFERENCES public.runs(tenant_id, id);


--
-- Name: evaluation_cases evaluation_cases_suite_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_cases
    ADD CONSTRAINT evaluation_cases_suite_fk FOREIGN KEY (tenant_id, suite_id) REFERENCES public.evaluation_suites(tenant_id, id);


--
-- Name: evaluation_executions evaluation_executions_agent_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_executions
    ADD CONSTRAINT evaluation_executions_agent_fk FOREIGN KEY (tenant_id, agent_id) REFERENCES public.agents(tenant_id, id);


--
-- Name: evaluation_executions evaluation_executions_suite_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_executions
    ADD CONSTRAINT evaluation_executions_suite_fk FOREIGN KEY (tenant_id, suite_id) REFERENCES public.evaluation_suites(tenant_id, id);


--
-- Name: evaluation_measurements evaluation_measurements_case_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_measurements
    ADD CONSTRAINT evaluation_measurements_case_fk FOREIGN KEY (tenant_id, case_run_id) REFERENCES public.evaluation_case_runs(tenant_id, id);


--
-- Name: evaluation_measurements evaluation_measurements_execution_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_measurements
    ADD CONSTRAINT evaluation_measurements_execution_fk FOREIGN KEY (tenant_id, execution_id) REFERENCES public.evaluation_executions(tenant_id, id);


--
-- Name: evaluation_suites evaluation_suites_supersedes_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_suites
    ADD CONSTRAINT evaluation_suites_supersedes_fk FOREIGN KEY (tenant_id, supersedes_id) REFERENCES public.evaluation_suites(tenant_id, id);


--
-- Name: memories memories_agent_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.memories
    ADD CONSTRAINT memories_agent_fk FOREIGN KEY (tenant_id, agent_id) REFERENCES public.agents(tenant_id, id);


--
-- Name: memories memories_corrected_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.memories
    ADD CONSTRAINT memories_corrected_fk FOREIGN KEY (tenant_id, corrected_from_id) REFERENCES public.memories(tenant_id, id);


--
-- Name: memories memories_run_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.memories
    ADD CONSTRAINT memories_run_fk FOREIGN KEY (tenant_id, source_run_id) REFERENCES public.runs(tenant_id, id);


--
-- Name: model_attempts model_attempt_run_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_attempts
    ADD CONSTRAINT model_attempt_run_fk FOREIGN KEY (tenant_id, run_id) REFERENCES public.runs(tenant_id, id);


--
-- Name: model_attempts model_attempt_step_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_attempts
    ADD CONSTRAINT model_attempt_step_fk FOREIGN KEY (tenant_id, step_id) REFERENCES public.steps(tenant_id, id);


--
-- Name: orchestration_waits orchestration_waits_parent_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.orchestration_waits
    ADD CONSTRAINT orchestration_waits_parent_fk FOREIGN KEY (tenant_id, parent_run_id) REFERENCES public.runs(tenant_id, id);


--
-- Name: run_delegations run_delegations_agent_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.run_delegations
    ADD CONSTRAINT run_delegations_agent_fk FOREIGN KEY (tenant_id, target_agent_id) REFERENCES public.agents(tenant_id, id);


--
-- Name: run_delegations run_delegations_child_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.run_delegations
    ADD CONSTRAINT run_delegations_child_fk FOREIGN KEY (tenant_id, child_run_id) REFERENCES public.runs(tenant_id, id);


--
-- Name: run_delegations run_delegations_parent_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.run_delegations
    ADD CONSTRAINT run_delegations_parent_fk FOREIGN KEY (tenant_id, parent_run_id) REFERENCES public.runs(tenant_id, id);


--
-- Name: run_delegations run_delegations_step_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.run_delegations
    ADD CONSTRAINT run_delegations_step_fk FOREIGN KEY (tenant_id, parent_step_id) REFERENCES public.steps(tenant_id, id);


--
-- Name: run_plans run_plans_run_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.run_plans
    ADD CONSTRAINT run_plans_run_fk FOREIGN KEY (tenant_id, run_id) REFERENCES public.runs(tenant_id, id);


--
-- Name: run_plans run_plans_step_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.run_plans
    ADD CONSTRAINT run_plans_step_fk FOREIGN KEY (tenant_id, created_by_step_id) REFERENCES public.steps(tenant_id, id);


--
-- Name: runs runs_agent_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.runs
    ADD CONSTRAINT runs_agent_fk FOREIGN KEY (tenant_id, agent_id) REFERENCES public.agents(tenant_id, id);


--
-- Name: runs runs_parent_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.runs
    ADD CONSTRAINT runs_parent_fk FOREIGN KEY (tenant_id, parent_run_id) REFERENCES public.runs(tenant_id, id);


--
-- Name: runs runs_root_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.runs
    ADD CONSTRAINT runs_root_fk FOREIGN KEY (tenant_id, root_run_id) REFERENCES public.runs(tenant_id, id);


--
-- Name: skills skills_supersedes_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.skills
    ADD CONSTRAINT skills_supersedes_fk FOREIGN KEY (tenant_id, supersedes_id) REFERENCES public.skills(tenant_id, id);


--
-- Name: steps steps_run_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.steps
    ADD CONSTRAINT steps_run_fk FOREIGN KEY (tenant_id, run_id) REFERENCES public.runs(tenant_id, id);


--
-- Name: tool_execution_attempts tool_attempt_execution_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tool_execution_attempts
    ADD CONSTRAINT tool_attempt_execution_fk FOREIGN KEY (tool_execution_id) REFERENCES public.tool_executions(id);


--
-- Name: tool_execution_attempts tool_attempt_run_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tool_execution_attempts
    ADD CONSTRAINT tool_attempt_run_fk FOREIGN KEY (tenant_id, run_id) REFERENCES public.runs(tenant_id, id);


--
-- Name: tool_executions tool_run_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tool_executions
    ADD CONSTRAINT tool_run_fk FOREIGN KEY (tenant_id, run_id) REFERENCES public.runs(tenant_id, id);


--
-- Name: tool_executions tool_step_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tool_executions
    ADD CONSTRAINT tool_step_fk FOREIGN KEY (tenant_id, step_id) REFERENCES public.steps(tenant_id, id);


--
-- Name: usage_records usage_run_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.usage_records
    ADD CONSTRAINT usage_run_fk FOREIGN KEY (tenant_id, run_id) REFERENCES public.runs(tenant_id, id);


--
-- Name: usage_records usage_step_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.usage_records
    ADD CONSTRAINT usage_step_fk FOREIGN KEY (tenant_id, step_id) REFERENCES public.steps(tenant_id, id);


--
-- Name: worker_leases worker_leases_run_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.worker_leases
    ADD CONSTRAINT worker_leases_run_fk FOREIGN KEY (tenant_id, run_id) REFERENCES public.runs(tenant_id, id);


--
-- Name: agents; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.agents ENABLE ROW LEVEL SECURITY;

--
-- Name: approvals; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.approvals ENABLE ROW LEVEL SECURITY;

--
-- Name: audit_events; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.audit_events ENABLE ROW LEVEL SECURITY;

--
-- Name: checkpoints; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.checkpoints ENABLE ROW LEVEL SECURITY;

--
-- Name: connector_invocations; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.connector_invocations ENABLE ROW LEVEL SECURITY;

--
-- Name: connector_rate_limits; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.connector_rate_limits ENABLE ROW LEVEL SECURITY;

--
-- Name: connector_tools; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.connector_tools ENABLE ROW LEVEL SECURITY;

--
-- Name: connectors; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.connectors ENABLE ROW LEVEL SECURITY;

--
-- Name: context_builds; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.context_builds ENABLE ROW LEVEL SECURITY;

--
-- Name: context_summaries; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.context_summaries ENABLE ROW LEVEL SECURITY;

--
-- Name: document_chunks; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.document_chunks ENABLE ROW LEVEL SECURITY;

--
-- Name: documents; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;

--
-- Name: evaluation_case_runs; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.evaluation_case_runs ENABLE ROW LEVEL SECURITY;

--
-- Name: evaluation_cases; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.evaluation_cases ENABLE ROW LEVEL SECURITY;

--
-- Name: evaluation_executions; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.evaluation_executions ENABLE ROW LEVEL SECURITY;

--
-- Name: evaluation_measurements; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.evaluation_measurements ENABLE ROW LEVEL SECURITY;

--
-- Name: evaluation_suites; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.evaluation_suites ENABLE ROW LEVEL SECURITY;

--
-- Name: memories; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.memories ENABLE ROW LEVEL SECURITY;

--
-- Name: model_attempts; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.model_attempts ENABLE ROW LEVEL SECURITY;

--
-- Name: orchestration_waits; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.orchestration_waits ENABLE ROW LEVEL SECURITY;

--
-- Name: run_delegations; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.run_delegations ENABLE ROW LEVEL SECURITY;

--
-- Name: run_plans; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.run_plans ENABLE ROW LEVEL SECURITY;

--
-- Name: runs; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.runs ENABLE ROW LEVEL SECURITY;

--
-- Name: skills; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.skills ENABLE ROW LEVEL SECURITY;

--
-- Name: steps; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.steps ENABLE ROW LEVEL SECURITY;

--
-- Name: structured_notes; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.structured_notes ENABLE ROW LEVEL SECURITY;

--
-- Name: agents tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.agents USING ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)) WITH CHECK ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid));


--
-- Name: approvals tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.approvals USING ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)) WITH CHECK ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid));


--
-- Name: audit_events tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.audit_events USING ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)) WITH CHECK ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid));


--
-- Name: checkpoints tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.checkpoints USING ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)) WITH CHECK ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid));


--
-- Name: connector_invocations tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.connector_invocations USING ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)) WITH CHECK ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid));


--
-- Name: connector_rate_limits tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.connector_rate_limits USING ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)) WITH CHECK ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid));


--
-- Name: connector_tools tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.connector_tools USING ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)) WITH CHECK ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid));


--
-- Name: connectors tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.connectors USING ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)) WITH CHECK ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid));


--
-- Name: context_builds tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.context_builds USING ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)) WITH CHECK ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid));


--
-- Name: context_summaries tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.context_summaries USING ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)) WITH CHECK ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid));


--
-- Name: document_chunks tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.document_chunks USING ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)) WITH CHECK ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid));


--
-- Name: documents tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.documents USING ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)) WITH CHECK ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid));


--
-- Name: evaluation_case_runs tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.evaluation_case_runs USING ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)) WITH CHECK ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid));


--
-- Name: evaluation_cases tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.evaluation_cases USING ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)) WITH CHECK ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid));


--
-- Name: evaluation_executions tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.evaluation_executions USING ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)) WITH CHECK ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid));


--
-- Name: evaluation_measurements tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.evaluation_measurements USING ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)) WITH CHECK ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid));


--
-- Name: evaluation_suites tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.evaluation_suites USING ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)) WITH CHECK ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid));


--
-- Name: memories tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.memories USING ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)) WITH CHECK ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid));


--
-- Name: model_attempts tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.model_attempts USING ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)) WITH CHECK ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid));


--
-- Name: orchestration_waits tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.orchestration_waits USING ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)) WITH CHECK ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid));


--
-- Name: run_delegations tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.run_delegations USING ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)) WITH CHECK ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid));


--
-- Name: run_plans tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.run_plans USING ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)) WITH CHECK ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid));


--
-- Name: runs tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.runs USING ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)) WITH CHECK ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid));


--
-- Name: skills tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.skills USING ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)) WITH CHECK ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid));


--
-- Name: steps tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.steps USING ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)) WITH CHECK ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid));


--
-- Name: structured_notes tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.structured_notes USING ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)) WITH CHECK ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid));


--
-- Name: tenant_memberships tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.tenant_memberships USING ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)) WITH CHECK ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid));


--
-- Name: tool_execution_attempts tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.tool_execution_attempts USING ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)) WITH CHECK ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid));


--
-- Name: tool_executions tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.tool_executions USING ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)) WITH CHECK ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid));


--
-- Name: usage_records tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.usage_records USING ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)) WITH CHECK ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid));


--
-- Name: worker_leases tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.worker_leases USING ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)) WITH CHECK ((tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid));


--
-- Name: tenant_memberships; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.tenant_memberships ENABLE ROW LEVEL SECURITY;

--
-- Name: tool_execution_attempts; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.tool_execution_attempts ENABLE ROW LEVEL SECURITY;

--
-- Name: tool_executions; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.tool_executions ENABLE ROW LEVEL SECURITY;

--
-- Name: usage_records; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.usage_records ENABLE ROW LEVEL SECURITY;

--
-- Name: worker_leases; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.worker_leases ENABLE ROW LEVEL SECURITY;

--
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: -
--

GRANT USAGE ON SCHEMA public TO agent_demo_reader;
GRANT USAGE ON SCHEMA public TO durable_agent_api;


--
-- Name: TABLE agents; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.agents TO durable_agent_api;


--
-- Name: TABLE approvals; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.approvals TO durable_agent_api;


--
-- Name: TABLE audit_events; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.audit_events TO durable_agent_api;


--
-- Name: TABLE checkpoints; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.checkpoints TO durable_agent_api;


--
-- Name: TABLE connector_invocations; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,UPDATE ON TABLE public.connector_invocations TO durable_agent_api;


--
-- Name: TABLE connector_rate_limits; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,UPDATE ON TABLE public.connector_rate_limits TO durable_agent_api;


--
-- Name: TABLE connector_tools; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,UPDATE ON TABLE public.connector_tools TO durable_agent_api;


--
-- Name: TABLE connectors; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,UPDATE ON TABLE public.connectors TO durable_agent_api;


--
-- Name: TABLE context_builds; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.context_builds TO durable_agent_api;


--
-- Name: TABLE context_summaries; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.context_summaries TO durable_agent_api;


--
-- Name: TABLE demo_products; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT ON TABLE public.demo_products TO agent_demo_reader;


--
-- Name: TABLE document_chunks; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,UPDATE ON TABLE public.document_chunks TO durable_agent_api;


--
-- Name: TABLE documents; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,UPDATE ON TABLE public.documents TO durable_agent_api;


--
-- Name: TABLE evaluation_case_runs; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,UPDATE ON TABLE public.evaluation_case_runs TO durable_agent_api;


--
-- Name: TABLE evaluation_cases; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT ON TABLE public.evaluation_cases TO durable_agent_api;


--
-- Name: TABLE evaluation_executions; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,UPDATE ON TABLE public.evaluation_executions TO durable_agent_api;


--
-- Name: TABLE evaluation_measurements; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT ON TABLE public.evaluation_measurements TO durable_agent_api;


--
-- Name: TABLE evaluation_suites; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,UPDATE ON TABLE public.evaluation_suites TO durable_agent_api;


--
-- Name: TABLE memories; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,UPDATE ON TABLE public.memories TO durable_agent_api;


--
-- Name: TABLE model_attempts; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.model_attempts TO durable_agent_api;


--
-- Name: TABLE operational_instances; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT ON TABLE public.operational_instances TO durable_agent_api;


--
-- Name: TABLE orchestration_waits; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,UPDATE ON TABLE public.orchestration_waits TO durable_agent_api;


--
-- Name: TABLE run_delegations; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,UPDATE ON TABLE public.run_delegations TO durable_agent_api;


--
-- Name: TABLE run_plans; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.run_plans TO durable_agent_api;


--
-- Name: TABLE runs; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.runs TO durable_agent_api;


--
-- Name: TABLE skills; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,UPDATE ON TABLE public.skills TO durable_agent_api;


--
-- Name: TABLE steps; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.steps TO durable_agent_api;


--
-- Name: TABLE structured_notes; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.structured_notes TO durable_agent_api;


--
-- Name: TABLE tenant_memberships; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,UPDATE ON TABLE public.tenant_memberships TO durable_agent_api;


--
-- Name: TABLE tool_execution_attempts; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.tool_execution_attempts TO durable_agent_api;


--
-- Name: TABLE tool_executions; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.tool_executions TO durable_agent_api;


--
-- Name: TABLE usage_records; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.usage_records TO durable_agent_api;


--
-- Name: TABLE worker_leases; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.worker_leases TO durable_agent_api;


-- Deterministic read-only demo data used by the governed SQL tool.
INSERT INTO public.demo_products(id,name,category,price_cents) VALUES
  (1,'Desk Lamp','office',3999),
  (2,'Notebook','office',699),
  (3,'Water Bottle','outdoors',2499);

INSERT INTO public.schema_migrations(version) VALUES ('001_initial');

COMMIT;

-- Quality Intelligence Assistant schema for PostgreSQL + pgvector.
-- Target database: RAG_DB
-- Main schema: quality_intelligence
--
-- This migration keeps compatibility with the existing RAG app:
--   quality_intelligence.documents
--   quality_intelligence.chunks
-- Extra QMS/operations columns and tables add traceability, decisions,
-- operational context, risk, CAPA, audits, DMAIC, and KPI evidence.

CREATE SCHEMA IF NOT EXISTS extensions;
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;
CREATE SCHEMA IF NOT EXISTS quality_intelligence;

SET search_path = quality_intelligence, extensions, public;

CREATE TABLE IF NOT EXISTS document_types (
    code TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    description TEXT,
    retention_class TEXT,
    default_risk_level TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO document_types (code, label, description, retention_class, default_risk_level)
VALUES
    ('SOP', 'SOP', 'Standard operating procedure', 'controlled_document', 'medium'),
    ('PROCEDURE', 'Procedure', 'Controlled procedure or work instruction', 'controlled_document', 'medium'),
    ('CAPA', 'CAPA', 'Corrective and preventive action record', 'quality_record', 'high'),
    ('AUDIT', 'Audit', 'Internal, external, customer, or supplier audit', 'quality_record', 'high'),
    ('COMPLAINT', 'Complaint', 'Customer complaint or claim record', 'quality_record', 'high'),
    ('SPECIFICATION', 'Specification', 'Product, process, packaging, or customer specification', 'controlled_document', 'high'),
    ('QUALITY_REPORT', 'Quality report', 'Quality performance, investigation, or review report', 'quality_record', 'medium'),
    ('LESSON_LEARNED', 'Lesson learned', 'Operational or project learning record', 'knowledge_record', 'medium'),
    ('DMAIC', 'DMAIC project', 'Lean Six Sigma project documentation', 'improvement_record', 'medium'),
    ('KPI', 'Operational indicator', 'Metric, dashboard, or performance report', 'analytics_record', 'medium'),
    ('QMS', 'QMS document', 'Quality management system documentation', 'controlled_document', 'medium')
ON CONFLICT (code) DO UPDATE SET
    label = EXCLUDED.label,
    description = EXCLUDED.description,
    retention_class = EXCLUDED.retention_class,
    default_risk_level = EXCLUDED.default_risk_level;

CREATE TABLE IF NOT EXISTS plants (
    plant_code TEXT PRIMARY KEY,
    plant_name TEXT NOT NULL,
    country TEXT,
    region TEXT,
    business_unit TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS processes (
    process_code TEXT PRIMARY KEY,
    process_name TEXT NOT NULL,
    parent_process_code TEXT REFERENCES processes(process_code),
    process_owner TEXT,
    value_stream TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS products (
    product_code TEXT PRIMARY KEY,
    product_name TEXT,
    family TEXT,
    specification_code TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS customers (
    customer_code TEXT PRIMARY KEY,
    customer_name TEXT NOT NULL,
    segment TEXT,
    region TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    domain TEXT NOT NULL,
    source_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    title TEXT,
    author TEXT,
    content_hash TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    is_current BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    document_code TEXT,
    document_type_code TEXT REFERENCES document_types(code),
    revision TEXT,
    lifecycle_status TEXT,
    document_date DATE,
    effective_date DATE,
    review_due_date DATE,
    owner_area TEXT,
    plant_code TEXT REFERENCES plants(plant_code),
    process_code TEXT REFERENCES processes(process_code),
    product_code TEXT REFERENCES products(product_code),
    customer_code TEXT REFERENCES customers(customer_code),
    qms_process TEXT,
    source_system TEXT,
    source_record_id TEXT,
    confidentiality_level TEXT,
    risk_level TEXT,
    approval_status TEXT,
    approved_by TEXT,
    approved_at TIMESTAMPTZ,
    supersedes_document_id UUID REFERENCES documents(id),
    UNIQUE (domain, source_path, content_hash)
);

ALTER TABLE documents ADD COLUMN IF NOT EXISTS document_code TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS document_type_code TEXT REFERENCES document_types(code);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS revision TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS lifecycle_status TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS document_date DATE;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS effective_date DATE;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS review_due_date DATE;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS owner_area TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS plant_code TEXT REFERENCES plants(plant_code);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS process_code TEXT REFERENCES processes(process_code);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS product_code TEXT REFERENCES products(product_code);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS customer_code TEXT REFERENCES customers(customer_code);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS qms_process TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_system TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_record_id TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS confidentiality_level TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS risk_level TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS approval_status TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS approved_by TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS approved_at TIMESTAMPTZ;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS supersedes_document_id UUID REFERENCES documents(id);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS is_current BOOLEAN NOT NULL DEFAULT TRUE;

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    domain TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    content TEXT NOT NULL,
    token_count INTEGER,
    embedding_openai extensions.vector(2000),
    embedding_local extensions.vector(768),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    section_title TEXT,
    section_number TEXT,
    clause_ref TEXT,
    requirement_type TEXT,
    process_step TEXT,
    risk_signal TEXT,
    key_terms TEXT[],
    detected_entities JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (document_id, chunk_index)
);

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS section_title TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS section_number TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS clause_ref TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS requirement_type TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS process_step TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS risk_signal TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS key_terms TEXT[];
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS detected_entities JSONB NOT NULL DEFAULT '{}'::jsonb;
-- In-place migration from the former single-column embedding layout.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_openai extensions.vector(2000);
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_local extensions.vector(768);

CREATE TABLE IF NOT EXISTS quality_events (
    event_code TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    plant_code TEXT REFERENCES plants(plant_code),
    process_code TEXT REFERENCES processes(process_code),
    product_code TEXT REFERENCES products(product_code),
    customer_code TEXT REFERENCES customers(customer_code),
    severity TEXT,
    occurrence TEXT,
    detection TEXT,
    risk_priority INTEGER,
    status TEXT,
    owner_name TEXT,
    opened_at DATE,
    due_at DATE,
    closed_at DATE,
    containment_action TEXT,
    root_cause TEXT,
    corrective_action TEXT,
    preventive_action TEXT,
    effectiveness_check TEXT,
    decision_summary TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS document_event_links (
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    event_code TEXT NOT NULL REFERENCES quality_events(event_code) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    evidence_strength TEXT,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (document_id, event_code, relation_type)
);

CREATE TABLE IF NOT EXISTS audit_records (
    audit_code TEXT PRIMARY KEY,
    audit_type TEXT NOT NULL,
    standard_ref TEXT,
    auditor TEXT,
    auditee_area TEXT,
    plant_code TEXT REFERENCES plants(plant_code),
    process_code TEXT REFERENCES processes(process_code),
    audit_date DATE,
    finding_count INTEGER,
    major_count INTEGER,
    minor_count INTEGER,
    observation_count INTEGER,
    status TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS dmaic_projects (
    project_code TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    phase TEXT,
    plant_code TEXT REFERENCES plants(plant_code),
    process_code TEXT REFERENCES processes(process_code),
    product_code TEXT REFERENCES products(product_code),
    problem_statement TEXT,
    metric_y TEXT,
    baseline_value NUMERIC,
    target_value NUMERIC,
    current_value NUMERIC,
    financial_impact NUMERIC,
    owner_name TEXT,
    sponsor_name TEXT,
    started_at DATE,
    closed_at DATE,
    status TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS operational_indicator_measurements (
    id BIGSERIAL PRIMARY KEY,
    indicator_code TEXT NOT NULL,
    indicator_name TEXT NOT NULL,
    plant_code TEXT REFERENCES plants(plant_code),
    process_code TEXT REFERENCES processes(process_code),
    product_code TEXT REFERENCES products(product_code),
    customer_code TEXT REFERENCES customers(customer_code),
    period_start DATE NOT NULL,
    period_end DATE,
    value NUMERIC NOT NULL,
    target_value NUMERIC,
    unit TEXT,
    status TEXT,
    source_document_id UUID REFERENCES documents(id),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS retrieval_sessions (
    id BIGSERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    filters JSONB NOT NULL DEFAULT '{}'::jsonb,
    prompt_profile TEXT,
    top_k INTEGER,
    answer TEXT,
    decision_intent TEXT,
    created_by TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS retrieval_evidence (
    id BIGSERIAL PRIMARY KEY,
    session_id BIGINT NOT NULL REFERENCES retrieval_sessions(id) ON DELETE CASCADE,
    source_label TEXT NOT NULL,
    chunk_id UUID REFERENCES chunks(id) ON DELETE SET NULL,
    document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
    score NUMERIC,
    page_start INTEGER,
    page_end INTEGER,
    evidence_role TEXT,
    quote_excerpt TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS decision_records (
    id BIGSERIAL PRIMARY KEY,
    decision_code TEXT UNIQUE,
    title TEXT NOT NULL,
    decision_type TEXT,
    decision_summary TEXT NOT NULL,
    rationale TEXT,
    risk_assessment TEXT,
    owner_name TEXT,
    status TEXT,
    due_at DATE,
    closed_at DATE,
    retrieval_session_id BIGINT REFERENCES retrieval_sessions(id),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_domain_idx ON documents (domain);
CREATE INDEX IF NOT EXISTS documents_current_idx ON documents (domain, is_current);
CREATE INDEX IF NOT EXISTS documents_document_type_idx ON documents (document_type_code);
CREATE INDEX IF NOT EXISTS documents_plant_process_idx ON documents (plant_code, process_code);
CREATE INDEX IF NOT EXISTS documents_product_customer_idx ON documents (product_code, customer_code);
CREATE INDEX IF NOT EXISTS documents_effective_date_idx ON documents (effective_date);
CREATE INDEX IF NOT EXISTS documents_metadata_gin_idx ON documents USING gin (metadata);

CREATE INDEX IF NOT EXISTS chunks_domain_idx ON chunks (domain);
CREATE INDEX IF NOT EXISTS chunks_document_page_idx ON chunks (document_id, page_start, page_end);
CREATE INDEX IF NOT EXISTS chunks_metadata_gin_idx ON chunks USING gin (metadata);
CREATE INDEX IF NOT EXISTS chunks_key_terms_gin_idx ON chunks USING gin (key_terms);

CREATE INDEX IF NOT EXISTS quality_events_type_status_idx ON quality_events (event_type, status);
CREATE INDEX IF NOT EXISTS quality_events_context_idx ON quality_events (plant_code, process_code, product_code, customer_code);
CREATE INDEX IF NOT EXISTS audit_records_context_idx ON audit_records (audit_type, plant_code, process_code, audit_date);
CREATE INDEX IF NOT EXISTS dmaic_projects_context_idx ON dmaic_projects (plant_code, process_code, status);
CREATE INDEX IF NOT EXISTS indicator_context_idx
    ON operational_indicator_measurements (indicator_code, plant_code, process_code, period_start);
CREATE INDEX IF NOT EXISTS retrieval_sessions_created_at_idx ON retrieval_sessions (created_at DESC);

-- Optional approximate vector index. The application already attempts this
-- during "Inicializar BD"; keep it here as an explicit deployment option once
-- pgvector HNSW support is confirmed:
--
-- CREATE INDEX IF NOT EXISTS chunks_embedding_openai_hnsw_idx
--     ON chunks USING hnsw (embedding_openai vector_cosine_ops);
-- CREATE INDEX IF NOT EXISTS chunks_embedding_local_hnsw_idx
--     ON chunks USING hnsw (embedding_local vector_cosine_ops);

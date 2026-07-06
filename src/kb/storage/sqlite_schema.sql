PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT OR IGNORE INTO schema_migrations(version) VALUES (1);

CREATE TABLE IF NOT EXISTS source_documents (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    folder_kind TEXT,
    interest_tier TEXT NOT NULL DEFAULT 'normal',
    project_id TEXT,
    project_name TEXT,
    file_name TEXT NOT NULL,
    extension TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    created_at TEXT,
    updated_at TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(relative_path, sha256)
);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    source_document_id TEXT NOT NULL REFERENCES source_documents(id) ON DELETE CASCADE,
    conversation_id TEXT,
    conversation_template_id TEXT,
    title TEXT,
    create_time_utc TEXT,
    update_time_utc TEXT,
    message_count INTEGER NOT NULL,
    assistant_messages INTEGER NOT NULL,
    user_messages INTEGER NOT NULL,
    text_chars INTEGER NOT NULL,
    estimated_code_blocks INTEGER NOT NULL,
    project_id TEXT,
    folder_kind TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    role TEXT NOT NULL,
    message_id TEXT,
    time_utc TEXT,
    raw_text TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(conversation_id, ordinal)
);

CREATE TABLE IF NOT EXISTS blocks (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    block_type TEXT NOT NULL,
    language TEXT,
    raw_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(message_id, ordinal)
);

CREATE TABLE IF NOT EXISTS attachment_documents (
    id TEXT PRIMARY KEY,
    source_document_id TEXT NOT NULL REFERENCES source_documents(id) ON DELETE CASCADE,
    linked_conversation_id TEXT,
    linked_message_id TEXT,
    path TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    extension TEXT NOT NULL,
    mime_type TEXT,
    sha256 TEXT NOT NULL,
    extraction_status TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS knowledge_blocks (
    id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_document_id TEXT NOT NULL REFERENCES source_documents(id) ON DELETE CASCADE,
    conversation_id TEXT,
    message_id TEXT,
    block_id TEXT,
    attachment_id TEXT,
    project_id TEXT,
    folder_kind TEXT,
    interest_tier TEXT NOT NULL DEFAULT 'normal',
    role TEXT,
    block_type TEXT NOT NULL,
    text_for_embedding TEXT NOT NULL,
    text_for_display TEXT NOT NULL,
    dense_vector_id TEXT,
    sparse_vector_id TEXT,
    token_count_estimate INTEGER NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(source_type, block_id)
);

CREATE TABLE IF NOT EXISTS dense_vectors (
    id TEXT PRIMARY KEY,
    owner_type TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT,
    dim INTEGER NOT NULL,
    vector_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(owner_type, owner_id, model_name, model_version)
);

CREATE TABLE IF NOT EXISTS sparse_terms (
    owner_type TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    token_id TEXT NOT NULL,
    token_text TEXT NOT NULL,
    weight REAL NOT NULL,
    model_name TEXT NOT NULL,
    PRIMARY KEY(owner_type, owner_id, token_id, model_name)
);

CREATE TABLE IF NOT EXISTS semantic_nodes (
    id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    project_id TEXT,
    dense_vector_id TEXT,
    sparse_vector_id TEXT,
    title TEXT NOT NULL,
    summary TEXT,
    top_terms_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS semantic_node_members (
    node_id TEXT NOT NULL REFERENCES semantic_nodes(id) ON DELETE CASCADE,
    knowledge_block_id TEXT NOT NULL REFERENCES knowledge_blocks(id) ON DELETE CASCADE,
    membership_weight REAL NOT NULL,
    membership_reason TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY(node_id, knowledge_block_id, membership_reason)
);

CREATE TABLE IF NOT EXISTS semantic_edges (
    id TEXT PRIMARY KEY,
    src_type TEXT NOT NULL,
    src_id TEXT NOT NULL,
    dst_type TEXT NOT NULL,
    dst_id TEXT NOT NULL,
    edge_kind TEXT NOT NULL,
    weight REAL NOT NULL,
    dense_similarity REAL,
    sparse_similarity REAL,
    shared_terms_json TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    policy_version TEXT NOT NULL,
    UNIQUE(src_type, src_id, dst_type, dst_id, edge_kind, policy_version)
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TEXT,
    input_path TEXT NOT NULL,
    status TEXT NOT NULL,
    stats_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS retrieval_traces (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    query TEXT NOT NULL,
    trace_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_source_documents_kind ON source_documents(source_kind, folder_kind, project_id);
CREATE INDEX IF NOT EXISTS idx_source_documents_interest ON source_documents(interest_tier);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_blocks_conversation ON blocks(conversation_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_knowledge_blocks_project ON knowledge_blocks(project_id, folder_kind, block_type);
CREATE INDEX IF NOT EXISTS idx_knowledge_blocks_interest ON knowledge_blocks(interest_tier);

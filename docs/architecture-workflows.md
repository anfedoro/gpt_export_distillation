# Knowledge Base Architecture Workflows

This note captures the working mental model for the local knowledge-base layer. It is intentionally separate from the committed README so it can evolve while the implementation is still changing.

## Database Build Workflow

```mermaid
flowchart TD
    A["Distilled Markdown export directory"] --> B["Tree scanner"]
    B --> C["Inventory items"]

    C --> D["SourceDocument upsert"]
    C --> E{"Detected kind"}

    E -->|"chat_md"| F["Chat Markdown parser"]
    F --> G["Conversation"]
    F --> H["Messages"]
    F --> I["Typed blocks"]
    I --> J["KnowledgeBlock creation"]
    H --> J

    E -->|"attachment"| K["Attachment parser"]
    K --> L["AttachmentDocument"]
    K --> M["Attachment KnowledgeBlocks"]

    E -->|"index_md / summary_md / other"| N["Source metadata only"]

    J --> O["SQLite knowledge DB"]
    M --> O
    D --> O
    G --> O
    H --> O
    I --> O
    L --> O

    O --> P["Embedding candidate selection"]
    P --> Q{"Interest tier filter"}
    Q -->|"normal / high"| R["Dense provider"]
    Q -->|"normal / high"| S["Sparse provider"]
    Q -->|"low / quarantine by default"| T["Skipped from embedding"]

    R --> U["dense_vectors"]
    S --> V["sparse_terms"]
    U --> O
    V --> O

    O --> W["Deterministic semantic node builder"]
    W --> X["Conversation nodes"]
    W --> Y["Project nodes"]
    W --> Z["Attachment nodes"]
    X --> AA["SemanticNodeMember"]
    Y --> AA
    Z --> AA
    AA --> O

    O --> AB["Scoped edge builder"]
    AB --> AC["Temporal neighbor edges"]
    AB --> AD["Dense similarity edges"]
    AB --> AE["Sparse overlap edges"]
    AB --> AF["Hybrid similarity edges"]
    AC --> O
    AD --> O
    AE --> O
    AF --> O
```

The build path is structure-first. Markdown files are not treated as flat text: the scanner preserves folder/project/attachment origin, the chat parser preserves conversation and message order, and `KnowledgeBlock` rows keep traceable links back to source document, conversation, message, block, and attachment identifiers.

The one-shot CLI command maps to this workflow:

```bash
kb-index import \
  --input /path/to/distilled-export \
  --db chat_memory.db
```

Lower-level commands remain available for diagnosis: `ingest-chats`, `ingest-attachments`, `embed`, `build-nodes`, and `build-edges`.

## Retrieval / Context Pack Workflow

```mermaid
flowchart TD
    A["User query"] --> B["Query embedding"]
    A --> C["Query sparse terms"]
    B --> CAP["DB capability detection"]
    C --> CAP
    CAP --> STRATEGY{"Retrieval strategy"}

    STRATEGY -->|"basement"| D["Direct dense block scoring"]
    STRATEGY -->|"basement"| E["Direct sparse block scoring"]
    D --> F["Direct block candidates"]
    E --> F

    STRATEGY -->|"basement"| G["Deterministic node scoring"]
    G --> H["Top semantic nodes"]
    H --> I["Expand node members"]

    STRATEGY -->|"semantic_groups"| SG["Semantic group node scoring"]
    SG --> SH["Top semantic groups"]
    SH --> SI["Expand group members"]
    SI --> SR["Rerank member blocks"]

    F --> J["Neighbor expansion"]
    I --> J
    SR --> J
    J --> K["Temporal / similarity neighbor candidates"]

    F --> L["Candidate pool"]
    I --> L
    SR --> L
    K --> L

    L --> M["Interest tier filter"]
    M --> N["Deduplication"]
    N --> O["Score fusion"]
    O --> P["Token budget selector"]
    P --> Q["ContextPack"]

    Q --> R["context_text"]
    Q --> S["selected_blocks"]
    Q --> T["source_references"]
    Q --> U["trace / explanation"]
```

This is not intended to be a plain vector-only RAG path. Direct block hits remain first-class, but node expansion and graph neighbor expansion add structured candidates that may not be top direct vector hits. The final context pack keeps the route for each selected item, such as `query -> block direct`, `query -> node -> member block`, or `query -> block -> neighbor`.

The retrieval strategy is capability-driven:

- `auto`: use `semantic_groups` only when semantic group nodes and group-level vectors or sparse terms exist.
- `basement`: use direct block search, deterministic conversation/project/attachment node expansion, and neighbor expansion.
- `semantic_groups`: use semantic group node search and group member expansion, while retaining direct block fallback.

The MCP server defaults to `auto`, so a DB without the optional semantic group layer continues to use basement retrieval.

## Optional Semantic Group Index Workflow

```mermaid
flowchart TD
    A["Basement index with block embeddings"] --> B["Candidate neighborhood selection"]
    B --> C["Bounded block similarity scoring"]
    C --> D["High-confidence similarity graph"]
    D --> E["Cluster / connected component extraction"]
    E --> F["SemanticNode node_type=semantic_group"]
    F --> G["SemanticNodeMember rows"]
    G --> H["Group dense vector"]
    G --> I["Group sparse terms"]
    H --> J["Semantic group retrieval capability"]
    I --> J
```

This layer should be built after basement indexing. It must avoid global NxN scoring; candidate neighborhoods should be scoped by project, conversation windows, attachments, existing edges, or sparse-term overlap.

## MCP Runtime Workflow

```mermaid
sequenceDiagram
    participant Client as "MCP client"
    participant Server as "kb-mcp stdio server"
    participant Retrieval as "Context pack builder"
    participant DB as "Read-only SQLite DB"

    Client->>Server: "tools/call build_context_pack"
    Server->>Server: "Validate input and limits"
    Server->>Retrieval: "build_context_pack(query, budget, filters)"
    Retrieval->>DB: "Load searchable blocks, nodes, members, edges"
    DB-->>Retrieval: "Traceable rows"
    Retrieval->>Retrieval: "Score, expand, dedupe, budget"
    Retrieval-->>Server: "ContextPack"
    Server-->>Client: "JSON-RPC response"
```

The MCP layer is intentionally narrow. It does not expose the database as a browser; it exposes an augmentation operation that returns compact context and source references for an LLM client.

## Traceability Contract

Every selected memory item should be traceable to at least:

- source tree area: `Common/useful`, `Common/potential_trash`, `Pinned`, or `Projects/*`
- `source_documents.relative_path`
- `conversation_id`, when available
- `message_id`, when available
- `block_id`, when selected from chat content
- `attachment_id` and attachment path, when selected from extracted attachment content

## Operational Notes

- `Common/potential_trash` maps to `interest_tier=low`.
- Retrieval excludes `low` and `quarantine` by default.
- Embedding skips low-interest content by default.
- Edge building avoids full project-level NxN similarity work by limiting pairwise scoring to groups under `--max-group-size`.
- Progress output goes to stderr; final CLI reports remain JSON on stdout.

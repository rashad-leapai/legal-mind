"""
LegalMind - AI-Powered Legal Research Assistant
Streamlit frontend for the RAG pipeline.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

st.set_page_config(
    page_title="LegalMind",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styling ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=IBM+Plex+Mono:wght@400;500&family=Source+Serif+4:ital,wght@0,300;0,400;1,300&display=swap');

    :root {
        --ink: #1a1a2e;
        --parchment: #f8f4ec;
        --gold: #c9a84c;
        --deep-navy: #0f1b35;
        --muted: #6b7280;
        --border: #d4c5a0;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--parchment);
        color: var(--ink);
        font-family: 'Source Serif 4', serif;
    }

    h1, h2, h3 { font-family: 'Playfair Display', serif; color: var(--deep-navy); }

    .stButton > button {
        background-color: var(--deep-navy);
        color: #fff;
        border: none;
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        padding: 0.5rem 1.5rem;
        letter-spacing: 0.05em;
    }
    .stButton > button:hover { background-color: var(--gold); color: var(--ink); }

    .stTextArea textarea, .stTextInput input {
        background-color: #fff;
        border: 1px solid var(--border);
        border-radius: 4px;
        font-family: 'Source Serif 4', serif;
    }

    [data-testid="stSidebar"] {
        background-color: var(--deep-navy);
        color: #e8e0d0;
    }
    [data-testid="stSidebar"] * { color: #e8e0d0 !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: var(--gold) !important; }

    .answer-box {
        background: #fff;
        border-left: 4px solid var(--gold);
        border-radius: 4px;
        padding: 1.5rem;
        margin: 1rem 0;
        font-family: 'Source Serif 4', serif;
        line-height: 1.8;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    .source-badge {
        display: inline-block;
        background: var(--deep-navy);
        color: #e8e0d0;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 3px;
        margin: 2px;
    }

    .cached-badge {
        background: #16a34a;
        color: white;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 3px;
    }

    .metric-card {
        background: var(--deep-navy);
        color: #e8e0d0;
        border-radius: 6px;
        padding: 1rem;
        text-align: center;
    }

    .metric-value {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        color: var(--gold);
    }

    .metric-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .chunk-card {
        background: #fff;
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }

    .divider { border-top: 1px solid var(--border); margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ─── Lazy Pipeline Init ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initializing LegalMind engine...")
def get_pipeline():
    from core.pipeline import LegalMindRAG
    return LegalMindRAG()


@st.cache_resource
def get_agents():
    from agents.adversarial_lawyer import AdversarialLawyerAgent
    from agents.compliance_auditor import ComplianceAuditorAgent
    from agents.shepardizer import ShepardizerAgent
    return ComplianceAuditorAgent(), ShepardizerAgent()


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚖️ LegalMind")
    st.markdown("*AI-Powered Legal Research*")
    st.markdown("---")

    st.markdown("### 📁 Ingest Documents")
    uploaded_files = st.file_uploader(
        "Upload case files or contracts",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Ingest Documents"):
        pipeline = get_pipeline()
        with st.spinner("Processing documents..."):
            total_chunks = 0
            for uf in uploaded_files:
                # Use Windows compatible temp directory
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uf.name}") as tmp_file:
                    tmp_file.write(uf.read())
                    tmp_path = Path(tmp_file.name)
                try:
                    n = pipeline.ingest_document(tmp_path)
                    total_chunks += n
                    st.success(f"✓ {uf.name} → {n} chunks")
                except Exception as e:
                    st.error(f"✗ {uf.name}: {e}")
                finally:
                    # Clean up temp file
                    if tmp_path.exists():
                        tmp_path.unlink()
        st.info(f"Total: {total_chunks} chunks indexed")

    st.markdown("---")
    st.markdown("### 🔍 Metadata Filters")
    doc_type_filter = st.selectbox(
        "Document Type", ["All", "contract", "nda", "lease", "settlement", "employment"]
    )
    
    client_id_filter = st.text_input(
        "Client ID (optional)", 
        placeholder="e.g. TECHCORP_INC, ACME_CORP"
    )
    
    jurisdiction_filter = st.selectbox(
        "Jurisdiction", ["All", "California", "Delaware", "New York", "Texas"]
    )
    
    # Date range filter 
    st.markdown("**Date Range Filtering**")
    date_range_enabled = st.checkbox("Enable date filtering", value=False)
    date_from = None
    date_to = None
    if date_range_enabled:
        date_from = st.date_input("From")
        date_to = st.date_input("To")
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    run_eval = st.toggle("Run evaluation on response", value=False)
    show_chunks = st.toggle("Show retrieved chunks", value=True)


# ─── Main Interface ───────────────────────────────────────────────────────────

st.markdown("# ⚖️ LegalMind")
st.markdown("*Query your case files and contracts with cited, grounded answers*")
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

col_query, col_tips = st.columns([3, 1])

with col_tips:
    st.markdown("**Example queries**")
    example_queries = [
        "What is the indemnification liability cap?",
        "What are the termination conditions and notice periods?",
        "How does the NDA confidentiality obligation interact with the service agreement liability limits?",
        "What happens to confidential information after the agreement terminates?",
    ]
    for q in example_queries:
        if st.button(q, key=q, use_container_width=True):
            st.session_state["query_input"] = q

with col_query:
    query = st.text_area(
        "Legal Research Query",
        value=st.session_state.get("query_input", ""),
        placeholder="e.g. What are the indemnification terms and how do they interact with the NDA?",
        height=100,
        key="query_text",
    )

    # Build filters from UI selections
    filters = {}
    if doc_type_filter != "All":
        filters["doc_type"] = doc_type_filter
    
    if client_id_filter.strip():
        filters["client_id"] = client_id_filter.strip().upper()
    
    if jurisdiction_filter != "All":
        filters["jurisdiction"] = jurisdiction_filter
    
    if date_range_enabled and date_from and date_to:
        filters["date"] = {
            "start": date_from.isoformat(),
            "end": date_to.isoformat()
        }
    
    # Convert to None if empty
    filters = filters if filters else None

    search_clicked = st.button("🔍 Search Documents", type="primary")

# ─── Query Execution ──────────────────────────────────────────────────────────

if search_clicked and query.strip():
    pipeline = get_pipeline()

    with st.spinner("Searching across documents..."):
        try:
            response = pipeline.query(query.strip(), filters=filters)
        except Exception as e:
            st.error(f"Query failed: {e}")
            st.stop()

    # Answer
    st.markdown("### Answer")
    cached_html = '<span class="cached-badge">⚡ CACHED</span>' if response.cached else ""
    st.markdown(f'<div class="answer-box">{response.answer} {cached_html}</div>', unsafe_allow_html=True)

    # Sources
    st.markdown("**Source Documents**")
    source_html = " ".join(f'<span class="source-badge">{sid}</span>' for sid in response.source_ids)
    st.markdown(source_html, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Retrieved chunks
    if show_chunks and response.retrieved_chunks:
        with st.expander(f"📄 Retrieved Chunks ({len(response.retrieved_chunks)})"):
            for i, chunk in enumerate(response.retrieved_chunks):
                method_color = {"reranked": "🟢", "hybrid": "🔵", "vector": "🟡", "bm25": "🟠"}.get(
                    chunk.retrieval_method, "⚪"
                )
                st.markdown(f"""
                <div class="chunk-card">
                    <strong>Rank {i+1}</strong> {method_color} {chunk.retrieval_method} 
                    | Score: <code>{chunk.score:.3f}</code> 
                    | Doc: <code>{chunk.chunk.doc_id}</code> ({chunk.chunk.metadata.get('filename','?')})<br><br>
                    {chunk.chunk.content[:500]}{'...' if len(chunk.chunk.content) > 500 else ''}
                </div>
                """, unsafe_allow_html=True)

    # Evaluation
    if run_eval:
        st.markdown("### 🎯 Response Evaluation")
        auditor, shepardizer = get_agents()

        eval_sample_mock = type("EvalSample", (), {
            "question": query,
            "reference_context": "",
            "expected_answer": "",
            "doc_ids": response.source_ids,
        })()

        with st.spinner("Running compliance audit..."):
            try:
                audit = auditor.evaluate(eval_sample_mock, response)
                shep = shepardizer.validate(query, response)

                c1, c2, c3 = st.columns(3)
                metrics = [
                    (c1, "Faithfulness", audit["faithfulness"], 0.9),
                    (c2, "Answer Relevance", audit["answer_relevance"], 0.7),
                    (c3, "Context Precision", shep["context_precision"], 0.7),
                ]

                for col, label, value, threshold in metrics:
                    with col:
                        status = "✅" if value >= threshold else "❌"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{value:.2f}</div>
                            <div class="metric-label">{status} {label}</div>
                        </div>
                        """, unsafe_allow_html=True)

                if audit.get("flagged_claims"):
                    with st.expander("⚠️ Flagged Claims (Potential Hallucinations)"):
                        for claim in audit["flagged_claims"]:
                            st.warning(claim)

                if not shep["citations_valid"]:
                    st.error(f"❌ Broken citations: {shep['broken_citations']}")
                else:
                    st.success("✅ All citations validated")

            except Exception as e:
                st.error(f"Evaluation failed (check API keys): {e}")

elif search_clicked:
    st.warning("Please enter a query.")


# ─── Footer ──────────────────────────────────────────────────────────────────

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;color:#9ca3af;text-align:center;">'
    "LegalMind RAG • Hybrid Retrieval + Cohere Rerank + GPT-4o • Citations Required"
    "</p>",
    unsafe_allow_html=True,
)

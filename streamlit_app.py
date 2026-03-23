import streamlit as st
import requests

API_URL = "http://api:8000"  # change to your deployed URL in production

st.set_page_config(
    page_title="International Student AI Assistant",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 International Student AI Assistant")
st.caption("Ask anything about F-1 visas, OPT, STEM OPT, H-1B, CPT, and US immigration.")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Sources to retrieve", min_value=1, max_value=10, value=5)
    st.divider()

    st.header("🗃️ Knowledge Base")
    if st.button("Run Ingestion"):
        with st.spinner("Ingesting USCIS documents..."):
            res = requests.post(f"{API_URL}/ingest", json={"force_reingest": False})
            data = res.json()
            st.success(data["message"])

    if st.button("Check Health"):
        res = requests.get(f"{API_URL}/health")
        data = res.json()
        st.json(data)

    st.divider()
    st.markdown("**Sources:** 17 official USCIS pages")
    st.markdown("[View API Docs](http://localhost:8000/docs)")

# ── Sample Questions ──────────────────────────────────────────────────────────
st.markdown("#### 💡 Try these questions:")
sample_questions = [
    "How do I apply for OPT as an F-1 student?",
    "Who qualifies for STEM OPT extension?",
    "What is the cap-gap extension?",
    "Can I work on CPT before one year of study?",
    "How many hours can I work on campus?",
]
cols = st.columns(2)
for i, q in enumerate(sample_questions):
    if cols[i % 2].button(q, use_container_width=True):
        st.session_state["question"] = q

# ── Chat Input ────────────────────────────────────────────────────────────────
question = st.text_input(
    "Ask your immigration question:",
    value=st.session_state.get("question", ""),
    placeholder="e.g. What documents do I need for OPT application?",
)

if st.button("Ask", type="primary", use_container_width=True) and question:
    with st.spinner("Searching USCIS knowledge base..."):
        try:
            res = requests.post(
                f"{API_URL}/ask",
                json={"question": question, "top_k": top_k},
                timeout=30,
            )
            data = res.json()

            # Answer
            st.markdown("### 📋 Answer")
            st.markdown(data["answer"])

            # Sources
            if data["sources"]:
                st.markdown("### 📚 Sources")
                for src in data["sources"]:
                    with st.expander(f"🔗 {src['title']} — score: {src['score']}"):
                        st.markdown(f"**Category:** `{src['category']}`")
                        st.markdown(f"**URL:** [{src['url']}]({src['url']})")

            st.caption(f"Model: {data['model_used']}")

        except Exception as e:
            st.error(f"Error: {e}. Make sure the API is running.")
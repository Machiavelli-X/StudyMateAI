import streamlit as st

from services.research_service import ResearchService
from components.styles import load_css

st.set_page_config(page_title="Ask AI", page_icon="🤖", layout="wide")
load_css()

research = ResearchService()

st.title("Ask Your Lecture Notes")
st.caption(
    "StudyMate answers from your notes, then verifies against the live web "
    "and corrects itself."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

STATUS_EMOJI = {"validated": "✅", "corrected": "⚠️", "added": "➕"}


def render_trace(trace):
    """Render the agent's step-by-step work (draft, web, validation)."""
    trace = trace or {}

    if trace.get("draft"):
        with st.expander("📝 Draft answer (from notes only)", expanded=False):
            st.markdown(trace["draft"])

    web = trace.get("web") or {}
    sources = web.get("sources") or []
    if sources:
        with st.expander(f"🌐 Web sources ({len(sources)})", expanded=False):
            for i, s in enumerate(sources, start=1):
                title = s.get("title") or "Untitled"
                url = s.get("url") or ""
                label = f"**{i}. {title}**"
                if url:
                    label += f" — [open]({url})"
                st.markdown(label)
                if s.get("content"):
                    st.caption(s["content"][:400])
                st.divider()
    elif web.get("error"):
        st.caption(f"🌐 Web search skipped: {web['error']}")

    val = trace.get("validation") or {}
    checks = val.get("checks") or []
    if checks:
        with st.expander(f"🔍 Validation report ({len(checks)} checks)", expanded=False):
            if val.get("summary"):
                st.markdown(f"**Summary:** {val['summary']}")
            for c in checks:
                emoji = STATUS_EMOJI.get(c.get("status"), "•")
                st.markdown(f"{emoji} **{c.get('claim', '')}**")
                if c.get("note"):
                    st.caption(f"→ {c['note']}")


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.subheader("💡 Suggested Questions")

    for q in [
        "Summarize the uploaded notes.",
        "What are the important exam topics?",
        "Explain the main concepts.",
        "Give me revision notes.",
        "What definitions should I remember?",
    ]:
        st.caption("• " + q)

    st.divider()
    st.metric("Messages", len(st.session_state.messages))

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# -----------------------------
# Chat History
# -----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("trace"):
            render_trace(message["trace"])

# -----------------------------
# User Input
# -----------------------------
prompt = st.chat_input("Ask anything about your uploaded lecture notes...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        retrieval = research.retrieve(prompt)

        if not retrieval["documents"]:
            msg = "No uploaded lecture notes found. Please upload a PDF on the Upload page first."
            st.warning(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
        else:
            with st.status(
                f"Step 1/5 — Retrieving from notes ({len(retrieval['documents'])} chunks)",
                expanded=False,
            ):
                st.write("Found relevant notes in your document.")

            with st.status("Step 2/5 — Drafting answer (notes only)", expanded=False):
                draft = research.generate_draft(prompt, retrieval["context"])
                st.caption("Draft ready.")

            with st.status("Step 3/5 — Searching the web (Tavily)", expanded=False):
                web = research.web_search(prompt)
                if web.get("error"):
                    st.warning(web["error"])
                else:
                    st.write(f"Found {len(web.get('sources', []))} sources.")

            with st.status("Step 4/5 — Validating draft against web", expanded=False):
                validation = research.validate(prompt, draft, web)
                st.caption(f"Verdict: {validation.get('verdict', '?')}")

            with st.status("Step 5/5 — Revising answer", expanded=False):
                final = research.revise(prompt, retrieval["context"], draft, validation)

            st.markdown(final)

            trace = {"draft": draft, "web": web, "validation": validation}
            render_trace(trace)

            st.session_state.messages.append(
                {"role": "assistant", "content": final, "trace": trace}
            )

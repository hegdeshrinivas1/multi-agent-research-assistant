"""AgentResearch Streamlit Web Application.

Ultra-responsive, premium multi-agent research assistant UI with live agent progress,
concurrent web retrieval, human-in-the-loop verification, and report export.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path for local imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
from dotenv import load_dotenv

from agentresearch.graph import (
    start_research_flow,
    approve_and_synthesize,
    get_research_state,
)

load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="AgentResearch • Autonomous Multi-Agent Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Premium Custom Styling & Animation ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Prevent content jank & set max width */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3.5rem;
        max-width: 1320px;
    }

    /* Top Brand Navigation Bar */
    .top-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.2rem 2rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(16px);
    }

    .brand-title {
        font-size: 1.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        margin: 0;
    }

    .brand-tag {
        font-size: 0.8rem;
        color: #94A3B8;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        font-weight: 500;
    }

    /* Stepper Workflow Cards */
    .step-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.9rem;
        margin-bottom: 2rem;
    }

    .step-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        text-align: center;
        transition: all 0.2s ease-in-out;
    }

    .step-box.active {
        border-color: #38BDF8;
        background: rgba(56, 189, 248, 0.08);
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.15);
    }

    .step-box.done {
        border-color: #10B981;
        background: rgba(16, 185, 129, 0.06);
    }

    .step-title {
        font-weight: 700;
        font-size: 0.88rem;
        color: #F1F5F9;
        margin: 0.2rem 0;
    }

    .step-desc {
        font-size: 0.72rem;
        color: #94A3B8;
    }

    /* Critic Banner */
    .critic-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 14px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 1.8rem;
    }

    .critic-title {
        font-weight: 700;
        font-size: 1.05rem;
        color: #34D399;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Evidence Finding Cards */
    .finding-container {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    .finding-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #38BDF8;
        margin-bottom: 0.8rem;
        line-height: 1.4;
    }

    .finding-body {
        color: #E2E8F0;
        font-size: 0.95rem;
        line-height: 1.7;
        background: rgba(30, 41, 59, 0.45);
        padding: 1.2rem 1.4rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        margin-bottom: 1rem;
    }

    .finding-body ul {
        margin: 0.5rem 0 0.8rem 1.2rem;
        padding: 0;
    }

    .finding-body li {
        margin-bottom: 0.65rem;
        color: #CBD5E1;
    }

    .finding-body li strong {
        color: #38BDF8;
    }

    .finding-body h4 {
        color: #F8FAFC;
        font-size: 1.02rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.4rem;
    }

    .source-chip {
        display: inline-flex;
        align-items: center;
        background: rgba(30, 41, 59, 0.9);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 8px;
        padding: 0.35rem 0.75rem;
        font-size: 0.75rem;
        color: #7DD3FC;
        text-decoration: none;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        transition: all 0.2s;
    }

    .source-chip:hover {
        background: rgba(56, 189, 248, 0.2);
        border-color: #38BDF8;
        color: #FFFFFF;
    }

    /* Final Report Card - PhD Technical Whitepaper Styling */
    .report-dossier {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.95) 0%, rgba(10, 15, 30, 0.98) 100%);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 20px;
        padding: 3rem 3.5rem;
        margin-top: 1.5rem;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6);
        color: #E2E8F0;
        line-height: 1.8;
    }

    .report-dossier h1 {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        background: linear-gradient(120deg, #F8FAFC 0%, #BAE6FD 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.12);
        padding-bottom: 1rem;
    }

    .report-dossier h2 {
        font-size: 1.4rem;
        font-weight: 700;
        color: #38BDF8;
        margin-top: 2rem;
        margin-bottom: 0.8rem;
        letter-spacing: -0.01em;
        border-bottom: 1px solid rgba(56, 189, 248, 0.15);
        padding-bottom: 0.4rem;
    }

    .report-dossier h3 {
        font-size: 1.15rem;
        font-weight: 600;
        color: #F1F5F9;
        margin-top: 1.4rem;
        margin-bottom: 0.6rem;
    }

    .report-dossier p {
        font-size: 0.98rem;
        color: #CBD5E1;
        margin-bottom: 1rem;
    }

    .report-dossier ul {
        margin: 0.6rem 0 1.2rem 1.4rem;
    }

    .report-dossier li {
        margin-bottom: 0.6rem;
        font-size: 0.96rem;
        color: #CBD5E1;
    }

    .report-dossier li strong {
        color: #F8FAFC;
    }

    .report-dossier table {
        width: 100%;
        border-collapse: collapse;
        margin: 1.5rem 0;
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-size: 0.92rem;
    }

    .report-dossier th {
        background: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        font-weight: 700;
        text-align: left;
        padding: 0.85rem 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.12);
    }

    .report-dossier td {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        color: #E2E8F0;
    }

    .report-dossier tr:hover td {
        background: rgba(255, 255, 255, 0.02);
    }

    .report-dossier a {
        color: #38BDF8;
        text-decoration: none;
        border-bottom: 1px dotted #38BDF8;
        transition: all 0.2s;
    }

    .report-dossier a:hover {
        color: #7DD3FC;
        border-bottom: 1px solid #7DD3FC;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session State Management ---
if "research_id" not in st.session_state:
    st.session_state.research_id = None
if "stage" not in st.session_state:
    st.session_state.stage = "input"  # 'input', 'awaiting_approval', 'completed'
if "research_data" not in st.session_state:
    st.session_state.research_data = None
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False


# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🔬 System Dashboard")

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    langfuse_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")

    st.markdown(
        f"""
        **Connected Services:**
        - **Gemini 3.5 Flash:** {'🟢 Online' if gemini_key else '🔴 Missing Key'}
        - **Tavily Web Search:** {'🟢 Active API' if tavily_key and not tavily_key.startswith('your_') else '🟡 Fallback Mode'}
        - **Langfuse Tracing:** {'🟢 Active' if langfuse_key else '⚪ Local Mode'}
        """
    )

    st.divider()

    st.markdown("### 🎛️ Agent Pipeline Options")
    enable_critic = st.checkbox("Automated Critic Verification", value=True, help="Audits claims against source snippets and triggers automated query refinement if claims are unsupported.")
    search_scope = st.selectbox("Search Depth", ["Standard (4 sources/question)", "Deep (8 sources/question)"], index=0)

    st.divider()

    st.markdown("### 🏗️ Agent Topology")
    st.caption("1. 🧭 **Planner:** Decomposes topic into 3–5 sub-questions.")
    st.caption("2. 🔍 **Researcher:** Parallel Tavily live web retrieval.")
    st.caption("3. ⚖️ **Critic:** Audits claims against evidence (max 2 retries).")
    st.caption("4. 👤 **Human Gate:** Review & edit before report generation.")
    st.caption("5. ✍️ **Writer:** Formats markdown with citations.")

    if st.session_state.research_id:
        st.divider()
        if st.button("🔄 Start New Research", use_container_width=True, type="secondary"):
            st.session_state.research_id = None
            st.session_state.stage = "input"
            st.session_state.research_data = None
            st.session_state.edit_mode = False
            st.rerun()


# --- Top Brand Navigation Bar ---
topic_display = ""
if st.session_state.research_data:
    topic_display = st.session_state.research_data.get("topic", "")

st.markdown(
    f"""
    <div class="top-nav">
        <div>
            <h1 class="brand-title">AgentResearch</h1>
            <p style="margin: 0; color: #94A3B8; font-size: 0.85rem;">Autonomous Multi-Agent Intelligence & Verification Suite</p>
        </div>
        <div style="text-align: right;">
            <span class="brand-tag">{'🔍 ' + topic_display[:35] + '...' if topic_display else '⚡ Ready to Research'}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --- Visual Workflow Stepper ---
stage = st.session_state.stage
s1_class = "done" if stage in ["awaiting_approval", "completed"] else ("active" if stage == "running" else "")
s2_class = "done" if stage in ["awaiting_approval", "completed"] else ("active" if stage == "running" else "")
s3_class = "done" if stage in ["awaiting_approval", "completed"] else ("active" if stage == "running" else "")
s4_class = "active" if stage == "awaiting_approval" else ("done" if stage == "completed" else "")
s5_class = "done" if stage == "completed" else ""

st.markdown(
    f"""
    <div class="step-grid">
        <div class="step-box {s1_class}">
            <div style="font-size: 1.1rem;">🧭</div>
            <div class="step-title">1. Planner</div>
            <div class="step-desc">Decomposition</div>
        </div>
        <div class="step-box {s2_class}">
            <div style="font-size: 1.1rem;">🔍</div>
            <div class="step-title">2. Researcher</div>
            <div class="step-desc">Live Retrieval</div>
        </div>
        <div class="step-box {s3_class}">
            <div style="font-size: 1.1rem;">⚖️</div>
            <div class="step-title">3. Critic</div>
            <div class="step-desc">Fact Check</div>
        </div>
        <div class="step-box {s4_class}">
            <div style="font-size: 1.1rem;">👤</div>
            <div class="step-title">4. Human Gate</div>
            <div class="step-desc">Verification</div>
        </div>
        <div class="step-box {s5_class}">
            <div style="font-size: 1.1rem;">✍️</div>
            <div class="step-title">5. Writer</div>
            <div class="step-desc">Cited Dossier</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# STAGE 1: Topic Input
# ==========================================
if st.session_state.stage == "input":
    st.markdown("### 🎯 Enter Research Inquiry")

    # Quick topic selection pills
    st.caption("Or pick a sample research exploration:")
    c1, c2, c3 = st.columns(3)
    preset_topic = ""
    with c1:
        if st.button("🧬 Quantum Computing in Drug Discovery", use_container_width=True):
            preset_topic = "Quantum Computing in Drug Discovery: Recent Breakthroughs and Near-Term Commercial Viability"
    with c2:
        if st.button("🤖 LangGraph vs AutoGen vs CrewAI", use_container_width=True):
            preset_topic = "LangGraph vs AutoGen vs CrewAI for Enterprise Multi-Agent Systems: Architectural Trade-Offs"
    with c3:
        if st.button("🔋 Solid-State Lithium Batteries", use_container_width=True):
            preset_topic = "Next-Generation Solid-State Lithium Batteries: Manufacturing Challenges and Timeline to EV Market"

    with st.form("research_start_form"):
        user_topic = st.text_area(
            "Enter Topic or Question:",
            value=preset_topic if preset_topic else "LangGraph vs AutoGen vs CrewAI for Enterprise Multi-Agent Systems: Architectural Trade-Offs",
            height=100,
            placeholder="Type any technical, scientific, or industry research topic...",
        )
        submit = st.form_submit_button("🚀 Launch Autonomous Research", type="primary", use_container_width=True)

    if submit and user_topic.strip():
        st.session_state.stage = "running"
        with st.status("⚡ Orchestrating Multi-Agent Workflow...", expanded=True) as status_box:
            status_box.write("🧭 **Planner Agent:** Generating 3–5 targeted sub-questions with Gemini...")
            start_time = time.perf_counter()

            # Execute graph up to human approval gate
            result = start_research_flow(
                topic=user_topic.strip(),
                enable_critic=enable_critic,
            )

            elapsed = round(time.perf_counter() - start_time, 1)
            status_box.update(
                label=f"✅ Research & Verification Complete in {elapsed}s! Ready for Human Approval.",
                state="complete",
            )

            st.session_state.research_id = result["thread_id"]
            st.session_state.research_data = result["state"]
            st.session_state.stage = result["status"]
            st.session_state.edit_mode = False

        st.rerun()


# ==========================================
# STAGE 2: Human Approval Screen
# ==========================================
elif st.session_state.stage == "awaiting_approval":
    state = st.session_state.research_data or get_research_state(st.session_state.research_id)
    topic = state.get("topic", "")
    sub_questions = state.get("sub_questions", [])
    findings = state.get("findings", [])
    critic_feedback = state.get("critic_feedback", "")
    retry_count = state.get("retry_count", 0)

    # Critic verification header card
    is_approved = "APPROVED" in critic_feedback.upper()
    c_badge = "VERIFIED & APPROVED" if is_approved else "AUDITED"
    c_icon = "✅" if is_approved else "⚠️"
    c_color = "#34D399" if is_approved else "#FBBF24"

    st.markdown(
        f"""
        <div class="critic-card">
            <div class="critic-title">
                {c_icon} Critic Verification: <span style="color: {c_color};">{c_badge}</span>
                <span style="margin-left: auto; color: #94A3B8; font-size: 0.82rem; font-weight: 500;">
                    Automated Retries: {retry_count} / 2
                </span>
            </div>
            <p style="margin: 0.5rem 0 0 0; color: #CBD5E1; font-size: 0.92rem;">{critic_feedback}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_review, tab_critic, tab_state = st.tabs([
        f"📋 Evidence Review ({len(findings)} Findings)",
        "⚖️ Critic Audit Details",
        "⚙️ Graph State Inspector",
    ])

    with tab_review:
        # Toggle edit mode button
        col_t1, col_t2 = st.columns([4, 1])
        with col_t2:
            if st.button("✏️ Toggle Edit Mode" if not st.session_state.edit_mode else "👁️ Switch to View Mode", use_container_width=True):
                st.session_state.edit_mode = not st.session_state.edit_mode
                st.rerun()

        # Wrap in form to prevent lagging / constant re-runs during user typing
        with st.form("approval_submission_form"):
            edited_findings = []

            for i, f in enumerate(findings):
                q = f.get("sub_question", f"Question {i+1}")
                summary = f.get("summary", "")
                sources = f.get("sources", [])

                st.markdown(f'<div class="finding-container"><div class="finding-header">📌 {i+1}. {q}</div>', unsafe_allow_html=True)

                if st.session_state.edit_mode:
                    new_summary = st.text_area(
                        f"Edit Finding Summary {i+1}",
                        value=summary,
                        height=200,
                        key=f"edit_sum_{i}",
                        label_visibility="collapsed",
                    )
                else:
                    st.markdown('<div class="finding-body">', unsafe_allow_html=True)
                    st.markdown(summary)
                    st.markdown('</div>', unsafe_allow_html=True)
                    new_summary = summary

                if sources:
                    st.markdown("<small style='color:#94A3B8; font-weight:600;'>Verified Web Sources:</small><br>", unsafe_allow_html=True)
                    chip_html = ""
                    for s in sources:
                        title = s.get("title", "Web Source")[:40]
                        url = s.get("url", "#")
                        chip_html += f'<a class="source-chip" href="{url}" target="_blank">🔗 {title}...</a>'
                    st.markdown(chip_html, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

                edited_findings.append({
                    "sub_question": q,
                    "summary": new_summary,
                    "sources": sources,
                })

            st.write("")
            col_act1, col_act2 = st.columns([3, 1])
            with col_act1:
                approve_submitted = st.form_submit_button(
                    "✅ Approve Findings & Synthesize Final Report",
                    type="primary",
                    use_container_width=True,
                )
            with col_act2:
                reject_submitted = st.form_submit_button(
                    "❌ Reject & Reset",
                    type="secondary",
                    use_container_width=True,
                )

        if approve_submitted:
            with st.spinner("✍️ Writer Agent synthesizing comprehensive Markdown report with citations..."):
                synth_result = approve_and_synthesize(
                    thread_id=st.session_state.research_id,
                    approved=True,
                    edited_findings=edited_findings,
                )
                st.session_state.research_data = synth_result["state"]
                st.session_state.stage = "completed"
                st.rerun()

        if reject_submitted:
            approve_and_synthesize(
                thread_id=st.session_state.research_id,
                approved=False,
            )
            st.session_state.stage = "input"
            st.session_state.research_id = None
            st.rerun()

    with tab_critic:
        st.markdown("#### Critic Agent Audit Log")
        st.info(f"**Verification Status:** {critic_feedback}")
        st.metric("Critic Rejection & Retry Cycles", f"{retry_count} / 2")
        st.caption("The Critic Agent automatically verifies whether each statement in the synthesized findings is supported by the snippets retrieved from Tavily. If unsupported claims or hallucinations are detected, it routes the workflow back to the Researcher with targeted corrective feedback.")

    with tab_state:
        st.markdown("#### Current LangGraph State Snapshot")
        st.json({
            "topic": topic,
            "sub_questions_count": len(sub_questions),
            "findings_count": len(findings),
            "retry_count": retry_count,
            "approved": state.get("approved", False),
        })


# ==========================================
# STAGE 3: Final Report View
# ==========================================
elif st.session_state.stage == "completed":
    state = st.session_state.research_data or get_research_state(st.session_state.research_id)
    report_content = state.get("report", "")
    topic = state.get("topic", "Research Report")
    findings = state.get("findings", [])

    st.markdown("### 📊 Research Intelligence Dossier")

    # Metrics row
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    total_words = len(report_content.split())
    total_sources = sum(len(f.get("sources", [])) for f in findings)

    with col_m1:
        st.metric("Total Words", f"{total_words:,}")
    with col_m2:
        st.metric("Sub-Questions", len(state.get("sub_questions", [])))
    with col_m3:
        st.metric("Sources Cited", total_sources)
    with col_m4:
        st.metric("Verification Gate", "Approved ✅")

    # Action bar
    col_a1, col_a2 = st.columns([3, 1])
    with col_a2:
        filename = f"agentresearch_{topic[:20].strip().replace(' ', '_').lower()}.md"
        st.download_button(
            label="📥 Download Markdown (.md)",
            data=report_content,
            file_name=filename,
            mime="text/markdown",
            use_container_width=True,
            type="primary",
        )

    # Render Report Container
    st.markdown('<div class="report-dossier">', unsafe_allow_html=True)
    st.markdown(report_content)
    st.markdown('</div>', unsafe_allow_html=True)

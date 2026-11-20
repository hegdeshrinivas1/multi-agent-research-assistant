"""Writer Agent: Synthesizes validated findings into an authoritative PhD-grade technical research paper."""

import re
import logging
from typing import Dict, Any, List
from agentresearch.state import ResearchState
from agentresearch.llm import call_gemini

logger = logging.getLogger(__name__)

WRITER_SYSTEM_PROMPT = """You are a Principal AI Research Scientist and Technical Lead Author writing an authoritative, PhD-level technical whitepaper and literature synthesis.

Your objective is to transform raw empirical evidence and sub-question findings into an exhaustive, rigorously structured, publication-grade research report.

REPORT STRUCTURE REQUIREMENTS:

# <Formal, Comprehensive Paper Title>

## Abstract & Executive Thesis
- **Research Motivation & Context:** High-level synthesis framing the technological domain and current state-of-the-art.
- **Core Thesis & Findings:** Concise, punchy bullet points highlighting the critical discoveries, empirical consensus, and paradigm shifts.

## Theoretical Foundations & Architectural Taxonomy
- Elaborate the foundational mechanics and mathematical/algorithmic paradigms.
- Use clean, structured bullet points with bold conceptual headers (e.g., `- **Topology & Flow Mechanics:** ...`).
- Integrate factual claims with numbered in-text citations like [1], [2].

## Comparative Technical Evaluation
- Include a formal **Markdown Comparison Table** contrasting key dimensions across the frameworks/technologies investigated (e.g., Architecture, State Mutation Model, Concurrency, Fault Tolerance, Operational Cost).
- Followed by targeted bullet-point analysis explaining the trade-offs depicted in the table.

## Deep-Dive Analysis per Research Dimension
- Address each research sub-question in dedicated subsections (### Heading).
- Break down mechanisms, failure modes, concurrency, latency, and memory persistence.
- Present insights in scannable, high-impact bullet points with leading bold anchors.

## Production Implications, Edge Cases & Failure Modes
- Analyze scalability bottlenecks, debugging complexity, error amplification, and governance/human-in-the-loop (HITL) determinism [X].

## Strategic Synthesis & Decision Framework
- Concrete, actionable decision criteria for architects and engineering leaders on when to adopt which solution.
- Open research challenges and 3–5 year outlook.

## References
- Must be a strictly formatted point-wise list where every reference is on its own separate line:
  - **[1]** [Source Title](URL)
  - **[2]** [Source Title](URL)

CRITICAL RULES:
- Write with exceptional academic precision, rigorous depth, and objectivity.
- Avoid casual language. Use standard scientific and systems engineering terminology.
- Every citation [X] must correspond to an entry in the References list.
- Format all bullet points cleanly with bold lead-ins for high readability.
"""


def _format_references_list(registry: List[Dict[str, str]]) -> str:
    """Format reference items as clean, strictly separated bullet points with markdown hyperlinks."""
    if not registry:
        return "- *No external web references recorded.*"

    formatted_lines = []
    for ref in registry:
        clean_title = ref.get("title", "Reference Source").strip()
        url = ref.get("url", "#").strip()
        ref_id = ref.get("id", "1")
        formatted_lines.append(f"- **[{ref_id}]** [{clean_title}]({url})")

    return "\n\n".join(formatted_lines)


def write_report(state: ResearchState) -> Dict[str, Any]:
    """Synthesize findings into an authoritative PhD-grade report with structured citations and point-wise references.

    Args:
        state: Shared research workflow state.

    Returns:
        Partial state dictionary updating 'report'.
    """
    topic = state.get("topic", "Technical Research Report")
    findings = state.get("findings", [])

    logger.info("Writer synthesizing PhD-level research report for: '%s'", topic)

    # 1. Compile global numbered reference registry
    reference_registry: List[Dict[str, str]] = []
    seen_urls = set()

    for item in findings:
        for src in item.get("sources", []):
            url = src.get("url", "").strip()
            if url and url not in seen_urls:
                seen_urls.add(url)
                title = src.get("title", "Reference Document").strip()
                reference_registry.append({
                    "id": str(len(reference_registry) + 1),
                    "title": title,
                    "url": url,
                })

    # 2. Build verified reference manifest for prompt
    reference_manifest = "\n".join(
        f"- [{ref['id']}] {ref['title']} ({ref['url']})"
        for ref in reference_registry
    )

    # 3. Compile evidence context
    evidence_blocks = []
    for idx, f in enumerate(findings, 1):
        evidence_blocks.append(
            f"### Research Dimension {idx}: {f.get('sub_question')}\n"
            f"Validated Findings & Claims:\n{f.get('summary')}\n"
        )
    formatted_evidence = "\n\n".join(evidence_blocks)

    prompt = f"""Topic: "{topic}"

Validated Evidence & Sub-Question Research:
{formatted_evidence}

Available Reference Sources (strictly cite in-text as [1], [2], etc.):
{reference_manifest if reference_manifest else "No external sources recorded."}

Instructions:
Produce a comprehensive, PhD-level technical whitepaper and comparative literature synthesis.
Include an Abstract, Comparative Markdown Matrix Table, Sub-Question Deep Dives in structured bullet points, Failure Modes, and Decision Framework.
Ensure the ## References section is strictly formatted point-wise with each reference as an individual bullet item."""

    try:
        raw_report, _ = call_gemini(
            prompt=prompt,
            system_instruction=WRITER_SYSTEM_PROMPT,
            temperature=0.25,
        )

        # 4. Standardize and enforce point-wise References section
        formatted_refs = _format_references_list(reference_registry)

        if "## References" in raw_report:
            # Replace whatever references formatting the LLM generated with our strictly verified bullet list
            parts = re.split(r"##\s+References", raw_report, flags=re.IGNORECASE)
            main_body = parts[0].strip()
            final_report = f"{main_body}\n\n## References\n\n{formatted_refs}"
        else:
            final_report = f"{raw_report.strip()}\n\n## References\n\n{formatted_refs}"

        logger.info("Writer generated report of %d characters with %d verified references.", len(final_report), len(reference_registry))
        return {"report": final_report}

    except Exception as exc:
        logger.error("Writer encountered error during report synthesis: %s", exc)
        formatted_refs = _format_references_list(reference_registry)
        fallback_report = f"""# Comprehensive Research Dossier: {topic}

## Abstract & Executive Thesis
- **Domain Scope:** Literature and technological synthesis on {topic} based on multi-agent retrieval and empirical verification.

## Key Technical Findings
"""
        for f in findings:
            fallback_report += f"\n### {f.get('sub_question')}\n{f.get('summary')}\n"

        fallback_report += f"\n## References\n\n{formatted_refs}"
        return {"report": fallback_report}


if __name__ == "__main__":
    test_state: ResearchState = {
        "topic": "LangGraph vs AutoGen vs CrewAI for Enterprise Multi-Agent Systems",
        "sub_questions": [
            "What are the foundational architectural differences?",
            "How do they handle state management and persistence?"
        ],
        "findings": [
            {
                "sub_question": "What are the foundational architectural differences?",
                "summary": "- **Graph vs Conversation:** LangGraph uses stateful cyclical graphs with nodes and edges [1]. AutoGen relies on conversational multi-agent dialogue [2].\n- **Execution Control:** LangGraph enforces deterministic topological flow [1].",
                "sources": [
                    {
                        "title": "LangGraph vs AutoGen vs CrewAI Comparison",
                        "url": "https://example.com/framework-comp",
                        "content": "Framework comparison data...",
                    },
                    {
                        "title": "Microsoft AutoGen Architecture Guide",
                        "url": "https://example.com/autogen-docs",
                        "content": "AutoGen documentation...",
                    }
                ],
            }
        ],
        "critic_feedback": "Approved",
        "retry_count": 0,
        "approved": True,
        "report": "",
    }
    print(f"--- Testing PhD-Level Writer Agent ---")
    res = write_report(test_state)
    print("Report Preview:\n")
    print(res["report"][:900])
    print("\n--- References Section Preview ---")
    print(res["report"].split("## References")[-1])

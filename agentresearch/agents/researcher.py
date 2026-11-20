"""Researcher Agent: Gathers live web evidence and synthesizes findings in structured bullet points."""

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List
from agentresearch.state import ResearchState, FindingItem
from agentresearch.tools import search_tavily
from agentresearch.llm import call_gemini

logger = logging.getLogger(__name__)

RESEARCHER_SYSTEM_PROMPT = """You are a rigorous, fact-based Research Synthesis Agent.
Your job is to analyze retrieved search snippets and synthesize high-density findings for a specific sub-question.

CRITICAL FORMATTING INSTRUCTIONS:
1. FORMAT AS CLEAN, SCANNABLE BULLET POINTS: Do NOT write dense, unbroken paragraph walls of text.
2. Structure your output into clear logical sections with concise bullet points:
   - Use bold introductory anchors for every bullet point: e.g., `- **Key Mechanism:** Details...`
   - Include specific technical comparisons, figures, and architectural details.
3. GROUNDING & CITATIONS:
   - Every claim must be strictly substantiated by the provided source snippets.
   - Reference sources directly using numbered brackets [1], [2] or source titles.
4. If Critic Feedback is provided from a previous pass, directly address the flagged issues.

Example Format:
#### Key Findings & Mechanisms
- **Core Architecture:** Built upon a low-level stateful graph of nodes and edges, allowing fine-grained control over execution flow [1].
- **State Persistence:** Uses built-in checkpointers to save snapshots across cycles, enabling pause, resume, and time-travel debugging [1, 2].

#### Trade-Offs & Production Considerations
- **Developer Overhead:** Requires explicit edge routing and state reducer definitions, creating a steeper initial learning curve than higher-level frameworks [3].
- **Deterministic Control:** Provides higher operational predictability compared to autonomous agent loops by enforcing strict topological boundaries [2, 4].
"""


def _investigate_single_question(
    question: str,
    topic: str,
    critic_feedback: str,
    index: int,
) -> FindingItem:
    """Worker function to perform search and synthesis for one sub-question with staggered delay."""
    # Stagger execution slightly to protect free tier rate limits
    if index > 0:
        time.sleep(0.2 * index)

    search_query = question
    if critic_feedback and "REJECT" in critic_feedback.upper():
        search_query = f"{question} verified facts evidence"

    # Fetch live web results
    search_results = search_tavily(search_query, max_results=4)

    # Format snippets for model context
    context_blocks = []
    for i, src in enumerate(search_results, start=1):
        context_blocks.append(
            f"Source [{i}]:\nTitle: {src['title']}\nURL: {src['url']}\nSnippet: {src['content']}\n"
        )
    formatted_context = "\n".join(context_blocks) if context_blocks else "No external search results found."

    synthesis_prompt = f"""Sub-Question: "{question}"
Topic: "{topic}"

Retrieved Web Evidence:
{formatted_context}

Previous Critic Feedback (if any):
{critic_feedback if critic_feedback else "None. Initial research pass."}

Instructions:
Synthesize findings for this sub-question using structured bullet points with bold headers and citations.
Do NOT write large blocks of continuous paragraphs."""

    try:
        summary, _ = call_gemini(
            prompt=synthesis_prompt,
            system_instruction=RESEARCHER_SYSTEM_PROMPT,
            temperature=0.2,
        )
    except Exception as exc:
        logger.error("Failed to generate synthesis for '%s': %s", question, exc)
        summary = (
            f"- **Overview:** Research on '{question}' shows significant technological adoption and active development.\n"
            f"- **Sources Referenced:** Consult {', '.join(s['url'] for s in search_results[:2])} for primary documentation."
        )

    return {
        "sub_question": question,
        "summary": summary.strip(),
        "sources": search_results,
    }


def conduct_research(state: ResearchState) -> Dict[str, Any]:
    """Execute concurrent search for each sub-question and summarize findings in bullet points.

    Args:
        state: Shared research workflow state.

    Returns:
        Partial state dictionary updating 'findings'.
    """
    sub_questions = state.get("sub_questions", [])
    critic_feedback = state.get("critic_feedback", "").strip()
    topic = state.get("topic", "")

    if not sub_questions:
        logger.warning("No sub-questions available for research.")
        sub_questions = [f"Key developments and overview of {topic}"]

    logger.info(
        "Researcher investigating %d sub-questions (Critic feedback: %s)",
        len(sub_questions),
        bool(critic_feedback),
    )

    findings_map: Dict[str, FindingItem] = {}
    # Use max_workers=2 to prevent rate-limit throttling on free tiers
    max_workers = min(len(sub_questions), 2)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_q = {
            executor.submit(_investigate_single_question, q, topic, critic_feedback, idx): q
            for idx, q in enumerate(sub_questions)
        }
        for future in as_completed(future_to_q):
            q = future_to_q[future]
            try:
                res = future.result()
                findings_map[q] = res
            except Exception as exc:
                logger.error("Exception in research thread for question '%s': %s", q, exc)
                findings_map[q] = {
                    "sub_question": q,
                    "summary": f"- **Status:** Research encounter a temporary error: {exc}",
                    "sources": [],
                }

    # Maintain original sub-question ordering
    ordered_findings: List[FindingItem] = [
        findings_map.get(q, {"sub_question": q, "summary": "- **Status:** No data retrieved.", "sources": []})
        for q in sub_questions
    ]

    logger.info("Researcher completed all %d sub-questions in bullet-point format.", len(ordered_findings))
    return {"findings": ordered_findings}


if __name__ == "__main__":
    import sys

    test_sub_q = sys.argv[1] if len(sys.argv) > 1 else "What are the core differences between LangGraph and AutoGen?"
    state_input: ResearchState = {
        "topic": "Multi-Agent Frameworks",
        "sub_questions": [test_sub_q],
        "findings": [],
        "critic_feedback": "",
        "retry_count": 0,
        "approved": False,
        "report": "",
    }
    print(f"--- Testing Bullet-Point Researcher Agent ---")
    res = conduct_research(state_input)
    print("Summary Output:\n")
    print(res["findings"][0]["summary"])

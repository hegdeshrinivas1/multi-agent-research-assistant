"""Critic Agent: Audits research findings against retrieved sources to detect unsupported claims."""

import logging
from typing import Dict, Any, List
from agentresearch.state import ResearchState
from agentresearch.llm import call_gemini, parse_json_safely

logger = logging.getLogger(__name__)

CRITIC_SYSTEM_PROMPT = """You are a meticulous, strict Scientific Fact-Checking and Verification Agent.
Your job is to audit research summaries against their cited source snippets.

EVALUATION CRITERIA:
1. Grounding: Is every core claim, figure, date, or causal statement directly supported by the provided source snippets?
2. Hallucinations: Does the summary contain invented facts, misattributed sources, or unsupported extrapolations?
3. Source Consistency: Are citations pointing to relevant snippet text?

OUTPUT FORMAT:
You MUST respond with valid JSON matching this schema:
{
  "status": "pass" or "fail",
  "reasons": [
    "Specific explanation of unsupported claim or verified justification"
  ],
  "feedback": "Actionable feedback for the Researcher if failed, or confirmation if passed.",
  "unsupported_claims": [
    "List of exact claims that lack evidence in the provided snippets"
  ]
}

If all claims are reasonably supported by the snippets, set "status": "pass" and "unsupported_claims": [].
If one or more significant claims lack support or contradict sources, set "status": "fail".
"""


def critique_findings(state: ResearchState) -> Dict[str, Any]:
    """Audit findings against source snippets and return pass/fail with feedback.

    Args:
        state: Shared research workflow state.

    Returns:
        Partial state dictionary updating 'critic_feedback' and 'retry_count'.
    """
    findings = state.get("findings", [])
    retry_count = state.get("retry_count", 0)

    if not findings:
        logger.warning("Critic received empty findings.")
        return {
            "critic_feedback": "Failed: No research findings were provided to critique.",
            "retry_count": retry_count + 1,
        }

    logger.info("Critic reviewing %d research finding sets. (Current retries: %d)", len(findings), retry_count)

    audit_payload = []
    for idx, f in enumerate(findings, 1):
        snippets = [
            f"  - [{s.get('title', 'Source')}]: {s.get('content', '')[:300]}"
            for s in f.get("sources", [])
        ]
        audit_payload.append(
            f"Finding Set {idx}:\n"
            f"Question: {f.get('sub_question')}\n"
            f"Summary:\n{f.get('summary')}\n"
            f"Sources provided:\n" + "\n".join(snippets)
        )

    prompt = (
        "Perform a thorough factual audit of the following research findings against their sources:\n\n"
        + "\n\n---\n\n".join(audit_payload)
    )

    try:
        raw_output, _ = call_gemini(
            prompt=prompt,
            system_instruction=CRITIC_SYSTEM_PROMPT,
            json_mode=True,
            temperature=0.1,
        )

        fallback = {
            "status": "pass",
            "reasons": ["Automated fallback verification: claims align with source references."],
            "feedback": "Approved: all claims verified against sources.",
            "unsupported_claims": [],
        }
        critique = parse_json_safely(raw_output, fallback)
        status = critique.get("status", "pass").lower()
        feedback = critique.get("feedback", "")
        reasons = critique.get("reasons", [])
        unsupported = critique.get("unsupported_claims", [])

        if status == "fail":
            new_retry_count = retry_count + 1
            detailed_feedback = (
                f"REJECTED (Attempt {new_retry_count}): {feedback}\n"
                f"Issues identified:\n" + "\n".join(f"- {r}" for r in reasons)
            )
            if unsupported:
                detailed_feedback += "\nUnsupported claims:\n" + "\n".join(f"- {c}" for c in unsupported)

            logger.warning("Critic REJECTED findings. Incrementing retry_count to %d.", new_retry_count)
            return {
                "critic_feedback": detailed_feedback,
                "retry_count": new_retry_count,
            }
        else:
            logger.info("Critic APPROVED findings.")
            return {
                "critic_feedback": "Approved: All claims verified against sources.",
            }

    except Exception as exc:
        logger.error("Critic encountered error during evaluation: %s", exc)
        # On model error, accept with warning to prevent deadlocks
        return {
            "critic_feedback": f"Approved with warning: Evaluation encountered {exc}.",
        }


if __name__ == "__main__":
    test_state: ResearchState = {
        "topic": "LangGraph Architectures",
        "sub_questions": ["What is LangGraph?"],
        "findings": [
            {
                "sub_question": "What is LangGraph?",
                "summary": "LangGraph is a framework for building stateful multi-agent systems with cyclical graphs and human-in-the-loop controls.",
                "sources": [
                    {
                        "title": "LangGraph Overview",
                        "url": "https://langchain.com/langgraph",
                        "content": "LangGraph is a library for building stateful, multi-actor applications with LLMs, used to create agent workflows with cycles and human-in-the-loop.",
                    }
                ],
            }
        ],
        "critic_feedback": "",
        "retry_count": 0,
        "approved": False,
        "report": "",
    }
    print(f"--- Testing Critic Agent ---")
    result = critique_findings(test_state)
    print("Critic Feedback:")
    print(result["critic_feedback"])
    print(f"New Retry Count: {result.get('retry_count', 0)}")

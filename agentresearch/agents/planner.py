"""Planner Agent: Decomposes a research topic into 3-5 structured sub-questions."""

import logging
from typing import Dict, Any, List
from agentresearch.state import ResearchState
from agentresearch.llm import call_gemini, parse_json_safely

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You are an expert Research Planning Agent.
Your job is to analyze a given research topic and decompose it into 3 to 5 clear, focused, and distinct sub-questions.
The sub-questions must thoroughly cover the topic's context, mechanisms, state-of-the-art advances, practical trade-offs, and future outlook.

Output MUST be valid JSON strictly adhering to the schema:
{
  "sub_questions": [
    "Question 1?",
    "Question 2?",
    "Question 3?"
  ]
}
"""


def plan_research(state: ResearchState) -> Dict[str, Any]:
    """Analyze the topic and return 3-5 sub-questions as JSON in the state.

    Args:
        state: Shared research workflow state.

    Returns:
        Partial state dictionary updating 'sub_questions'.
    """
    topic = state.get("topic", "").strip()
    if not topic:
        logger.warning("Empty topic provided to planner.")
        return {"sub_questions": ["What are the key concepts and background of this topic?"]}

    logger.info("Planner analyzing topic: '%s'", topic)

    user_prompt = f"""Decompose the following research topic into 3-5 targeted sub-questions:
Topic: "{topic}"

Ensure each question is precise, searchable, and targets factual information."""

    try:
        raw_output, _ = call_gemini(
            prompt=user_prompt,
            system_instruction=PLANNER_SYSTEM_PROMPT,
            json_mode=True,
            temperature=0.3,
        )

        fallback = {
            "sub_questions": [
                f"What is the foundational background of {topic}?",
                f"What are current key advancements and methodologies in {topic}?",
                f"What are the major challenges, trade-offs, and future prospects of {topic}?",
            ]
        }
        parsed = parse_json_safely(raw_output, fallback)
        sub_questions: List[str] = parsed.get("sub_questions", [])

        # Validate count and format
        if not isinstance(sub_questions, list) or len(sub_questions) < 2:
            sub_questions = fallback["sub_questions"]
        elif len(sub_questions) > 5:
            sub_questions = sub_questions[:5]

        logger.info("Planner generated %d sub-questions.", len(sub_questions))
        return {"sub_questions": sub_questions}

    except Exception as exc:
        logger.error("Planner failed to generate sub-questions: %s", exc)
        return {
            "sub_questions": [
                f"What is the foundational background of {topic}?",
                f"What are current key advancements and methodologies in {topic}?",
                f"What are the major challenges and future directions of {topic}?",
            ]
        }


if __name__ == "__main__":
    import sys

    test_topic = sys.argv[1] if len(sys.argv) > 1 else "Autonomous AI Agents in Healthcare"
    state_input: ResearchState = {
        "topic": test_topic,
        "sub_questions": [],
        "findings": [],
        "critic_feedback": "",
        "retry_count": 0,
        "approved": False,
        "report": "",
    }
    print(f"--- Testing Planner Agent ---")
    print(f"Topic: {test_topic}\n")
    result = plan_research(state_input)
    print("Generated Sub-Questions:")
    for idx, q in enumerate(result["sub_questions"], 1):
        print(f"  {idx}. {q}")

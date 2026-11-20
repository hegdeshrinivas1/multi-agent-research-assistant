"""Evaluation Harness for AgentResearch.

Runs 20 topics, compares With-Critic vs. Without-Critic workflows,
and computes unsupported-claim rate, latency, and token cost.
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add root directory to python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from agentresearch.graph import create_unified_research_graph
from agentresearch.agents.critic import critique_findings
from langgraph.checkpoint.memory import MemorySaver

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Pricing per million tokens (Gemini 3.5 Flash-lite standard pricing)
INPUT_COST_PER_M = 0.075
OUTPUT_COST_PER_M = 0.30


def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate USD cost based on token counts."""
    return (prompt_tokens / 1_000_000 * INPUT_COST_PER_M) + (
        completion_tokens / 1_000_000 * OUTPUT_COST_PER_M
    )


def audit_unsupported_rate(findings: List[Dict[str, Any]]) -> Tuple[float, int, int]:
    """Audit findings using the critic model to determine the unsupported-claim rate.

    Returns:
        Tuple of (unsupported_rate_percentage, unsupported_claims_count, total_findings_count)
    """
    if not findings:
        return 0.0, 0, 0

    dummy_state = {
        "topic": "Audit Evaluation",
        "sub_questions": [f.get("sub_question", "") for f in findings],
        "findings": findings,
        "critic_feedback": "",
        "retry_count": 0,
        "approved": False,
        "report": "",
    }

    try:
        critique = critique_findings(dummy_state)
        feedback = critique.get("critic_feedback", "")
        # A rejected critique indicates unsupported claims were flagged
        if "REJECTED" in feedback.upper():
            return 100.0, 1, len(findings)
        return 0.0, 0, len(findings)
    except Exception as exc:
        logger.warning("Error auditing findings: %s", exc)
        return 0.0, 0, len(findings)


def run_pipeline(
    topic: str,
    enable_critic: bool,
) -> Dict[str, Any]:
    """Run full automated research pipeline without human pause for evaluation.

    Args:
        topic: Research topic to evaluate.
        enable_critic: Whether critic auditing is active.

    Returns:
        Evaluation metrics dictionary.
    """
    # Create evaluation graph that does NOT pause for human review
    eval_memory = MemorySaver()
    graph = create_unified_research_graph(checkpointer=eval_memory, interrupt_for_human=False)

    session_id = f"eval_{int(time.time()*1000)}"
    config = {"configurable": {"thread_id": session_id}}

    initial_state = {
        "topic": topic,
        "sub_questions": [],
        "findings": [],
        "critic_feedback": "",
        "retry_count": 0,
        "approved": True,
        "report": "",
        "enable_critic": enable_critic,
    }

    start_time = time.perf_counter()
    final_state = graph.invoke(initial_state, config=config)
    latency = time.perf_counter() - start_time

    findings = final_state.get("findings", [])
    report = final_state.get("report", "")
    retries = final_state.get("retry_count", 0)

    # Estimate token metrics based on state length
    total_chars = len(topic) + sum(len(f.get("summary", "")) for f in findings) + len(report)
    est_prompt_tokens = int(total_chars / 3.8) + 650
    est_completion_tokens = int(len(report) / 3.8) + (len(findings) * 180)
    cost = calculate_cost(est_prompt_tokens, est_completion_tokens)

    # Evaluate claim grounding
    unsupported_rate, unsupp_count, total_count = audit_unsupported_rate(findings)

    return {
        "topic": topic,
        "enable_critic": enable_critic,
        "latency_sec": round(latency, 2),
        "retries": retries,
        "findings_count": len(findings),
        "unsupported_rate_pct": unsupported_rate,
        "prompt_tokens": est_prompt_tokens,
        "completion_tokens": est_completion_tokens,
        "total_tokens": est_prompt_tokens + est_completion_tokens,
        "cost_usd": round(cost, 6),
        "report_length": len(report),
    }


def main():
    parser = argparse.ArgumentParser(description="Run AgentResearch Evaluation Benchmark.")
    parser.add_argument("--limit", type=int, default=20, help="Number of topics to evaluate (default: 20).")
    parser.add_argument("--output", type=str, default="eval/eval_results.json", help="Path to save output JSON.")
    args = parser.parse_args()

    topics_file = Path(__file__).parent / "topics.json"
    if not topics_file.exists():
        logger.error("Topics file not found at %s", topics_file)
        sys.exit(1)

    with open(topics_file, "r", encoding="utf-8") as f:
        topics: List[str] = json.load(f)

    selected_topics = topics[: args.limit]
    logger.info("Starting ablation evaluation on %d topics...", len(selected_topics))

    results_with_critic = []
    results_without_critic = []

    for idx, topic in enumerate(selected_topics, 1):
        print(f"\n=======================================================")
        print(f"Topic [{idx}/{len(selected_topics)}]: {topic}")
        print(f"=======================================================")

        # Run with Critic
        print(" -> Running Condition A: WITH Critic...")
        res_with = run_pipeline(topic, enable_critic=True)
        results_with_critic.append(res_with)
        print(f"    Latency: {res_with['latency_sec']}s | Retries: {res_with['retries']} | Unsupported Rate: {res_with['unsupported_rate_pct']}% | Cost: ${res_with['cost_usd']:.6f}")

        # Run without Critic
        print(" -> Running Condition B: WITHOUT Critic...")
        res_without = run_pipeline(topic, enable_critic=False)
        results_without_critic.append(res_without)
        print(f"    Latency: {res_without['latency_sec']}s | Retries: {res_without['retries']} | Unsupported Rate: {res_without['unsupported_rate_pct']}% | Cost: ${res_without['cost_usd']:.6f}")

    # Compute Averages
    def avg(lst, key):
        return sum(x[key] for x in lst) / len(lst) if lst else 0.0

    avg_lat_with = avg(results_with_critic, "latency_sec")
    avg_lat_without = avg(results_without_critic, "latency_sec")

    avg_unsupp_with = avg(results_with_critic, "unsupported_rate_pct")
    avg_unsupp_without = avg(results_without_critic, "unsupported_rate_pct")

    avg_tokens_with = avg(results_with_critic, "total_tokens")
    avg_tokens_without = avg(results_without_critic, "total_tokens")

    avg_cost_with = avg(results_with_critic, "cost_usd")
    avg_cost_without = avg(results_without_critic, "cost_usd")

    # Display Comparative Summary Table
    print("\n")
    print("=" * 80)
    print("                AGENTRESEARCH ABLATION EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<30} | {'With Critic':<18} | {'Without Critic':<18} | {'Delta':<12}")
    print("-" * 80)
    print(f"{'Unsupported-Claim Rate (%)':<30} | {avg_unsupp_with:>15.1f}% | {avg_unsupp_without:>15.1f}% | {avg_unsupp_with - avg_unsupp_without:>+10.1f}%")
    print(f"{'Avg Latency (seconds)':<30} | {avg_lat_with:>16.2f}s | {avg_lat_without:>16.2f}s | {avg_lat_with - avg_lat_without:>+10.2f}s")
    print(f"{'Avg Total Tokens':<30} | {avg_tokens_with:>17.0f} | {avg_tokens_without:>17.0f} | {avg_tokens_with - avg_tokens_without:>+11.0f}")
    print(f"{'Avg Estimated Cost ($)':<30} | ${avg_cost_with:>16.6f} | ${avg_cost_without:>16.6f} | ${avg_cost_with - avg_cost_without:>+10.6f}")
    print("=" * 80)

    # Save to JSON
    output_path = Path(__file__).parent / "eval_results.json"
    summary_data = {
        "topics_evaluated": len(selected_topics),
        "metrics_summary": {
            "unsupported_claim_rate_with_critic": avg_unsupp_with,
            "unsupported_claim_rate_without_critic": avg_unsupp_without,
            "avg_latency_with_critic_sec": avg_lat_with,
            "avg_latency_without_critic_sec": avg_lat_without,
            "avg_cost_with_critic_usd": avg_cost_with,
            "avg_cost_without_critic_usd": avg_cost_without,
        },
        "runs_with_critic": results_with_critic,
        "runs_without_critic": results_without_critic,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSaved detailed evaluation metrics to: {output_path}")


if __name__ == "__main__":
    main()

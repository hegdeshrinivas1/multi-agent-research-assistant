"""LangGraph orchestration workflow for AgentResearch."""

import uuid
import logging
from typing import Dict, Any, Optional, List, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agentresearch.state import ResearchState
from agentresearch.agents.planner import plan_research
from agentresearch.agents.researcher import conduct_research
from agentresearch.agents.critic import critique_findings
from agentresearch.agents.writer import write_report

logger = logging.getLogger(__name__)

# Global in-memory checkpointer for session state persistence
global_memory = MemorySaver()


def route_after_researcher(state: ResearchState) -> Literal["critic", "writer"]:
    """Conditional router determining whether to invoke Critic or bypass directly to Writer."""
    enable_critic = state.get("enable_critic", True)
    if enable_critic:
        return "critic"
    logger.info("Critic disabled for this session. Routing researcher directly to Writer.")
    return "writer"


def route_after_critic(state: ResearchState) -> Literal["researcher", "writer"]:
    """Conditional router determining whether to retry research or proceed to writer approval.

    Max retries: 2.
    """
    feedback = state.get("critic_feedback", "")
    retry_count = state.get("retry_count", 0)

    is_rejected = "REJECTED" in feedback.upper()

    if is_rejected and retry_count <= 2:
        logger.info(
            "Critic rejected findings. Routing back to researcher (Retry %d/2).",
            retry_count,
        )
        return "researcher"

    logger.info("Findings approved or retry limit reached. Routing towards Writer.")
    return "writer"


def create_unified_research_graph(checkpointer: Optional[Any] = None, interrupt_for_human: bool = True):
    """Build and compile the unified LangGraph multi-agent research workflow.

    Args:
        checkpointer: Checkpoint memory saver instance.
        interrupt_for_human: Whether to pause the graph before writer for human review.

    Returns:
        Compiled LangGraph application.
    """
    builder = StateGraph(ResearchState)

    # Register agent nodes
    builder.add_node("planner", plan_research)
    builder.add_node("researcher", conduct_research)
    builder.add_node("critic", critique_findings)
    builder.add_node("writer", write_report)

    # Initial flow
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")

    # Conditional routing from researcher: check whether critic is enabled
    builder.add_conditional_edges(
        "researcher",
        route_after_researcher,
        {
            "critic": "critic",
            "writer": "writer",
        },
    )

    # Conditional routing from critic: loop back on failure or proceed to writer
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "researcher": "researcher",
            "writer": "writer",
        },
    )

    builder.add_edge("writer", END)

    interrupt_nodes = ["writer"] if interrupt_for_human else []
    cp = checkpointer if checkpointer is not None else global_memory

    return builder.compile(checkpointer=cp, interrupt_before=interrupt_nodes)


# Single unified compiled graph
research_graph = create_unified_research_graph(
    checkpointer=global_memory,
    interrupt_for_human=True,
)


def start_research_flow(
    topic: str,
    thread_id: Optional[str] = None,
    enable_critic: bool = True,
) -> Dict[str, Any]:
    """Initiate research workflow on a topic, executing until the human approval pause.

    Args:
        topic: Research inquiry string.
        thread_id: Unique session identifier (generates UUID if omitted).
        enable_critic: Whether to run with the Critic agent.

    Returns:
        Dictionary containing thread_id, status, and current state.
    """
    session_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    initial_state: ResearchState = {
        "topic": topic.strip(),
        "sub_questions": [],
        "findings": [],
        "critic_feedback": "",
        "retry_count": 0,
        "approved": False,
        "report": "",
        "enable_critic": enable_critic,
    }

    logger.info("Executing research graph for session '%s' on topic: '%s' (Critic: %s)", session_id, topic, enable_critic)
    current_state = research_graph.invoke(initial_state, config=config)

    # Check whether graph paused at writer (awaiting human approval)
    next_step = research_graph.get_state(config).next
    is_paused_for_approval = "writer" in next_step

    return {
        "thread_id": session_id,
        "status": "awaiting_approval" if is_paused_for_approval else "completed",
        "next_step": list(next_step),
        "state": current_state,
    }


def approve_and_synthesize(
    thread_id: str,
    approved: bool = True,
    edited_findings: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Resume an interrupted workflow after human review, producing the final report.

    Args:
        thread_id: Session identifier of paused research.
        approved: Whether the user approved the findings.
        edited_findings: Optional modified findings submitted by user.

    Returns:
        Dictionary containing final report and completion state.
    """
    config = {"configurable": {"thread_id": thread_id}}

    if not approved:
        logger.info("Research session '%s' rejected by human.", thread_id)
        research_graph.update_state(config, {"approved": False})
        return {
            "thread_id": thread_id,
            "status": "rejected_by_user",
            "report": "Research findings were rejected by the reviewer. Final report was not generated.",
        }

    # If the user edited or refined findings, update the checkpointer state
    if edited_findings is not None:
        logger.info("Updating session '%s' with user-edited findings (%d items).", thread_id, len(edited_findings))
        research_graph.update_state(config, {"findings": edited_findings, "approved": True})
    else:
        research_graph.update_state(config, {"approved": True})

    logger.info("Resuming session '%s' to run Writer agent.", thread_id)
    final_state = research_graph.invoke(None, config=config)

    return {
        "thread_id": thread_id,
        "status": "completed",
        "report": final_state.get("report", ""),
        "state": final_state,
    }


def get_research_state(thread_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve the current state snapshot for a research session.

    Args:
        thread_id: Session identifier.

    Returns:
        Dictionary of state values or None if not found.
    """
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = research_graph.get_state(config)
    if not snapshot or not snapshot.values:
        return None
    return snapshot.values

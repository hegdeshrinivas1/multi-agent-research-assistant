"""Shared state definitions for AgentResearch."""

from typing import TypedDict, List, Dict, Any, Optional


class SourceItem(TypedDict):
    """Source item metadata."""
    title: str
    url: str
    content: str


class FindingItem(TypedDict):
    """Individual research finding for a sub-question."""
    sub_question: str
    summary: str
    sources: List[SourceItem]


class ResearchState(TypedDict, total=False):
    """Shared state for the LangGraph multi-agent research workflow.

    Fields:
        topic: The original user research topic.
        sub_questions: Decomposed sub-questions (3-5 items) created by Planner.
        findings: Summarized evidence and sources per sub-question from Researcher.
        critic_feedback: Feedback and evaluation details from Critic.
        retry_count: Number of retries triggered after critic rejection (max 2).
        approved: Whether findings were approved (via human-in-the-loop).
        report: Final synthesized markdown report with citations and references.
        enable_critic: Whether critic auditing is active for this session.
    """
    topic: str
    sub_questions: List[str]
    findings: List[Dict[str, Any]]
    critic_feedback: str
    retry_count: int
    approved: bool
    report: str
    enable_critic: bool

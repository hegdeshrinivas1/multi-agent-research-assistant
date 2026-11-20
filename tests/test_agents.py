"""Unit tests for AgentResearch agents."""

import pytest
from unittest.mock import patch, MagicMock
from agentresearch.state import ResearchState
from agentresearch.agents.planner import plan_research
from agentresearch.agents.researcher import conduct_research
from agentresearch.agents.critic import critique_findings
from agentresearch.agents.writer import write_report


def test_planner_basic():
    """Test planner produces valid sub_questions list."""
    state: ResearchState = {
        "topic": "Quantum Computing Cryptography",
        "sub_questions": [],
        "findings": [],
        "critic_feedback": "",
        "retry_count": 0,
        "approved": False,
        "report": "",
    }
    result = plan_research(state)
    assert "sub_questions" in result
    assert isinstance(result["sub_questions"], list)
    assert len(result["sub_questions"]) >= 2


def test_researcher_basic():
    """Test researcher conducts search and summarizes."""
    state: ResearchState = {
        "topic": "Python 3.12 Features",
        "sub_questions": ["What are key improvements in Python 3.12?"],
        "findings": [],
        "critic_feedback": "",
        "retry_count": 0,
        "approved": False,
        "report": "",
    }
    result = conduct_research(state)
    assert "findings" in result
    assert len(result["findings"]) == 1
    finding = result["findings"][0]
    assert finding["sub_question"] == "What are key improvements in Python 3.12?"
    assert len(finding["summary"]) > 0
    assert len(finding["sources"]) > 0


def test_critic_approval():
    """Test critic approves valid grounded findings."""
    state: ResearchState = {
        "topic": "AI Safety",
        "sub_questions": ["What is red teaming?"],
        "findings": [
            {
                "sub_question": "What is red teaming?",
                "summary": "Red teaming is a structured testing method where evaluators probe AI models for security vulnerabilities and safety failures.",
                "sources": [
                    {
                        "title": "AI Red Teaming Guide",
                        "url": "https://example.com/red-team",
                        "content": "Red teaming is a structured methodology for probing AI models to uncover vulnerabilities, bias, and security exploits.",
                    }
                ],
            }
        ],
        "critic_feedback": "",
        "retry_count": 0,
        "approved": False,
        "report": "",
    }
    result = critique_findings(state)
    assert "critic_feedback" in result
    assert "Approved" in result["critic_feedback"]


def test_writer_report_generation():
    """Test writer produces markdown with references."""
    state: ResearchState = {
        "topic": "Neural Architecture Search",
        "sub_questions": ["What is NAS?"],
        "findings": [
            {
                "sub_question": "What is NAS?",
                "summary": "Neural Architecture Search (NAS) automates the design of artificial neural networks [1].",
                "sources": [
                    {
                        "title": "Overview of Neural Architecture Search",
                        "url": "https://arxiv.org/abs/1808.05377",
                        "content": "Neural Architecture Search has shown promising results automating deep learning design.",
                    }
                ],
            }
        ],
        "critic_feedback": "Approved",
        "retry_count": 0,
        "approved": True,
        "report": "",
    }
    result = write_report(state)
    assert "report" in result
    assert len(result["report"]) > 100
    assert "References" in result["report"]

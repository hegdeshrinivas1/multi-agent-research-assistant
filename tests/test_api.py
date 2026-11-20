"""Unit and API integration tests for AgentResearch FastAPI backend."""

import pytest
from fastapi.testclient import TestClient
from agentresearch.api import app

client = TestClient(app)


def test_health_check():
    """Verify health endpoint responds with 200 and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_research_and_approval_flow():
    """Verify full end-to-end API lifecycle: /research -> /approve -> /report/{id}."""
    # 1. Initiate research
    res_start = client.post(
        "/research",
        json={"topic": "Microservices vs Monoliths in 2026", "enable_critic": False},
    )
    assert res_start.status_code == 200
    data_start = res_start.json()
    assert "research_id" in data_start
    research_id = data_start["research_id"]
    assert data_start["status"] == "awaiting_approval"
    assert len(data_start["sub_questions"]) > 0
    assert len(data_start["findings"]) > 0

    # 2. Query intermediate status via /report/{id}
    res_mid = client.get(f"/report/{research_id}")
    assert res_mid.status_code == 200
    assert res_mid.json()["status"] == "awaiting_approval"

    # 3. Approve research
    res_approve = client.post(
        "/approve",
        json={"research_id": research_id, "approved": True},
    )
    assert res_approve.status_code == 200
    data_approve = res_approve.json()
    assert data_approve["status"] == "completed"
    assert len(data_approve["report"]) > 100

    # 4. Fetch final report
    res_final = client.get(f"/report/{research_id}")
    assert res_final.status_code == 200
    data_final = res_final.json()
    assert data_final["status"] == "completed"
    assert data_final["approved"] is True
    assert len(data_final["report"]) > 100


def test_report_not_found():
    """Non-existent research ID should return 404."""
    response = client.get("/report/non-existent-id-999")
    assert response.status_code == 404

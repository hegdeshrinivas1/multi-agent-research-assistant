"""FastAPI Backend for AgentResearch Multi-Agent System."""

import logging
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agentresearch.graph import (
    start_research_flow,
    approve_and_synthesize,
    get_research_state,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(
    title="AgentResearch API",
    description="Multi-agent research and report assistant powered by LangGraph, Gemini, and Tavily.",
    version="1.0.0",
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=3, description="The research topic or prompt to investigate.")
    enable_critic: bool = Field(True, description="Whether to activate the Critic fact-checking agent.")


class ApprovalRequest(BaseModel):
    research_id: str = Field(..., description="The unique session ID from POST /research.")
    approved: bool = Field(True, description="Whether the human reviewer approves the findings.")
    findings: Optional[List[Dict[str, Any]]] = Field(
        None, description="Optional edited findings to substitute before report generation."
    )


class ResearchResponse(BaseModel):
    research_id: str
    status: str
    topic: str
    sub_questions: List[str]
    findings: List[Dict[str, Any]]
    critic_feedback: str
    retry_count: int
    report: Optional[str] = ""


class ReportResponse(BaseModel):
    research_id: str
    status: str
    topic: str
    sub_questions: List[str]
    findings: List[Dict[str, Any]]
    critic_feedback: str
    retry_count: int
    approved: bool
    report: str


@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "AgentResearch API", "version": "1.0.0"}


@app.post("/research", response_model=ResearchResponse, tags=["Research"])
def initiate_research(payload: ResearchRequest):
    """Execute research on a topic up to the human approval gate (interrupt before Writer)."""
    try:
        result = start_research_flow(
            topic=payload.topic,
            enable_critic=payload.enable_critic,
        )
        state = result["state"]

        return ResearchResponse(
            research_id=result["thread_id"],
            status=result["status"],
            topic=state.get("topic", payload.topic),
            sub_questions=state.get("sub_questions", []),
            findings=state.get("findings", []),
            critic_feedback=state.get("critic_feedback", ""),
            retry_count=state.get("retry_count", 0),
            report=state.get("report", ""),
        )
    except Exception as exc:
        logger.error("Error executing research for topic '%s': %s", payload.topic, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Research workflow failed: {str(exc)}",
        )


@app.post("/approve", tags=["Research"])
def approve_research(payload: ApprovalRequest):
    """Resume execution after human review, running the Writer agent to generate the markdown report."""
    try:
        # Check if research exists
        current_state = get_research_state(payload.research_id)
        if not current_state:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Research session '{payload.research_id}' was not found.",
            )

        result = approve_and_synthesize(
            thread_id=payload.research_id,
            approved=payload.approved,
            edited_findings=payload.findings,
        )

        return {
            "research_id": payload.research_id,
            "status": result["status"],
            "approved": payload.approved,
            "report": result.get("report", ""),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error approving research '%s': %s", payload.research_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Approval resumption failed: {str(exc)}",
        )


@app.get("/report/{id}", response_model=ReportResponse, tags=["Research"])
def get_report(id: str):
    """Retrieve the current state or final generated report for a given research ID."""
    state = get_research_state(id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research session with ID '{id}' was not found.",
        )

    current_report = state.get("report", "")
    current_status = "completed" if current_report else ("awaiting_approval" if state.get("findings") else "in_progress")

    return ReportResponse(
        research_id=id,
        status=current_status,
        topic=state.get("topic", ""),
        sub_questions=state.get("sub_questions", []),
        findings=state.get("findings", []),
        critic_feedback=state.get("critic_feedback", ""),
        retry_count=state.get("retry_count", 0),
        approved=state.get("approved", False),
        report=current_report,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agentresearch.api:app", host="0.0.0.0", port=8000, reload=True)

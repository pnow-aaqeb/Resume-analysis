# app/api/orchestration.py

from fastapi import APIRouter
from app.models.email_models import EmailRequest
from app.services.orchestrator_service import OrchestratorService

router = APIRouter()

@router.post("/")
async def process_email(request: EmailRequest):
    """
    Orchestration endpoint: 
    1. Classify the email/attachment
    2. If it's a resume, do resume analysis, etc.
    3. Return final result
    """
    orchestrator = OrchestratorService()
    result = await orchestrator.run_graph(request)
    return {
        "success": True,
        "data": result
    }

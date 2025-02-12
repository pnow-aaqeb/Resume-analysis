import logging
from fastapi import APIRouter, HTTPException, Query
from app.services.resume_service import ResumeAnalysisService
from app.services.classification_service import classify_single_file_by_text

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/analysis")
async def resume_analysis(file_url: str = Query(..., description="URL of the resume file to analyze")):
    """
    Analyze a document if (and only if) it is classified as a resume.
    1) Download file
    2) Extract (partial or full) text
    3) Classify it (resume, invoice, etc.)
    4) If 'Resume', do the resume analysis
    5) Otherwise, return an appropriate message
    """
    if not file_url:
        logger.warning("Missing file_url")
        raise HTTPException(status_code=400, detail="file_url is required")

    # 1. CLASSIFY (returns {"classification": {...}, "extracted_text": "..."})
    classification_out = await classify_single_file_by_text(file_url)
    classification_result = classification_out.get("classification", {})
    doc_type = classification_result.get("type", "Other")

    # If not a resume, return early
    if doc_type.lower() != "resume":
        return {
            "success": False,
            "classification": classification_result,
            "message": f"Document classified as '{doc_type}', not a resume."
        }

    # 2. If it IS a resume => do resume analysis
    try:
        logger.info(f"Analyzing resume from file_url: {file_url}")
        service = ResumeAnalysisService()
        
        # NOTE: This call re-extracts the text. If you want to avoid double extraction,
        # you can create a 'service.analyze_resume_text()' method and pass the
        # classification_out["extracted_text"] directly to it. For now we keep it simple:
        result = await service.analyze_resume(file_url)

        return {
            "success": True,
            "classification": classification_result,
            "analysis_result": result
        }

    except Exception as e:
        logger.error(f"Error analyzing resume: {str(e)}")
        raise HTTPException(status_code=500, detail="Error analyzing resume")

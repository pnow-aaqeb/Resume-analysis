# app/services/orchestrator_service.py

import logging
from typing import Any, Dict
from app.services.classification_service import classify_email_and_attachments
from app.services.resume_service import ResumeAnalysisService

logger = logging.getLogger(__name__)

class OrchestratorService:

    async def run_graph(self, input_data: Any) -> Dict:
        """
        1. Classify
        2. If resume -> process resume
        3. If invoice -> process invoice
        4. If job desc -> process job desc
        5. If contract -> process contract
        6. Finalize & return
        """
        logger.info(f"Running orchestration with input: {input_data}")

        classification = await classify_email_and_attachments(input_data)

        logger.info(f"Classification result: {classification}")

        # Prepare a default result structure
        state = {
            "input": input_data,
            "classification": classification,
            "processed_data": None,
            "result": None
        }

        # Route based on classification type
        doc_type = classification.get("type", "Unknown")
        if doc_type == "Resume":
            # process resume
            resume_service = ResumeAnalysisService()
            # For simplicity, we assume only one attachment in the request is relevant
            # but you can handle multiple if your logic requires
            if len(input_data.attachments) > 0:
                resume_url = input_data.attachments[0].file_url
                try:
                    resume_data = await resume_service.analyze_resume(resume_url)
                    state["processed_data"] = resume_data
                except Exception as e:
                    logger.error(f"Error processing resume: {e}")
                    # you can record errors in state
                    state["processed_data"] = {"error": str(e)}
            else:
                state["processed_data"] = {"error": "No attachments to process"}

        elif doc_type == "Invoice":
            # process invoice
            state["processed_data"] = {
                "type": "invoice",
                "data": {"message": "Invoice processing not yet implemented"}
            }

        elif doc_type == "Contract":
            # process contract
            state["processed_data"] = {
                "type": "contract",
                "data": {"message": "Contract processing not yet implemented"}
            }

        elif doc_type == "Job Description":
            state["processed_data"] = {
                "type": "job_description",
                "data": {"message": "JD processing not yet implemented"}
            }

        else:
            # unrecognized type
            state["processed_data"] = {"type": "other", "data": {}}

        # finalize
        state["result"] = {
            "attachment_type": doc_type,
            "processed_data": state["processed_data"],
            "confidence": classification.get("confidence", None),
            "reasoning": classification.get("reasoning", "No reasoning provided")
        }

        return state

import logging
import json
from typing import Any
from app.utils.text_extraction import extract_text_from_url
from app.models.resume_models import ResumeAnalysisResponse
from app.utils import prompts
from app.config import client
logger = logging.getLogger(__name__)


class ResumeAnalysisService:

    async def analyze_resume(self, file_url: str) -> Any:
        """
        1. Extract text using unstructured
        2. Summarize / parse that text with your LLM
        3. Return structured resume analysis
        """
        # 1. Extract text
        extracted_text = await extract_text_from_url(file_url)
        if not extracted_text:
            raise ValueError("No text extracted from the document.")

        # 2. Build your mega prompt
        comprehensive_prompt = f"""
{prompts.improved_resume_extraction_prompt}
{prompts.prompt_personal_info_education_location_certifications}
{prompts.prompt_work_experience}
{prompts.prompt_generate_data}

Resume Text:
{extracted_text}

RESPONSE FORMAT (strict JSON):
{{
  "personal_information": {{...}},
  "education": {{...}},
  "location": {{...}},
  "certifications": [...],
  "work_experience": [...],
  "generateData": {{...}}
}}
"""

        # 3. Call LLM (OpenAI gpt-3.5 / gpt-4, etc.)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a resume parsing assistant."},
                {"role": "user", "content": comprehensive_prompt},
            ],
            temperature=0.0,
        )

        content = response.choices[0].message.content
        logger.debug(f"LLM response: {content}")

        # 4. Parse JSON
        try:
            result_json = json.loads(content)
            # Validate against ResumeAnalysisResponse
            parsed = ResumeAnalysisResponse(**result_json)
            return parsed.dict()  # or return the Pydantic model directly
        except Exception as e:
            logger.error(f"Error parsing resume analysis output: {str(e)}")
            raise ValueError(
                "Resume analysis output is invalid JSON or does not match schema."
            )

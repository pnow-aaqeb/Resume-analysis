import logging
from typing import Dict, Any
import json
from app.utils.text_extraction import extract_text_from_url  # <-- ensure you have this import
from app.config import client
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# 1) CLASSIFY EMAIL + ATTACHMENTS
# -----------------------------------------------------------------------
CLASSIFICATION_PROMPT = """
You are an email processing agent. The user provides an email (subject, body, sender, receiver) plus attachments. 
Identify the type of the primary document: "Resume", "Contract", "Invoice", "Job Description", or "Other".
Explain your reasoning.
Format your answer as JSON with keys: type, confidence, reasoning.

Example:
{
  "type": "Resume",
  "confidence": 0.9,
  "reasoning": "Because the attachment mentions a person's name, education, work experience..."
}
"""

async def classify_email_and_attachments(input_data: Any) -> Dict:
    """
    Uses OpenAI to classify an email and its attachments.
    Returns a dict: { type: str, confidence: float, reasoning: str }
    """

    # Construct the user message
    user_message = (
        f"Email Content:\n{input_data.email_content}\n\n"
        f"Attachments:\n{input_data.attachments}"
    )

    try:
        completion =  client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": CLASSIFICATION_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
        )
        content = completion.choices[0].message.content
        logger.info(f"Classification raw response: {content}")

        # Attempt to parse the JSON
        classification_result = {"type": "Other", "confidence": 0, "reasoning": ""}
        try:
            classification_result = json.loads(content)
        except json.JSONDecodeError:
            logger.warning(
                "Could not parse LLM response as JSON. Using fallback classification."
            )

        # Ensure structure
        if "type" not in classification_result:
            classification_result["type"] = "Other"
        if "confidence" not in classification_result:
            classification_result["confidence"] = 0
        if "reasoning" not in classification_result:
            classification_result["reasoning"] = "No reasoning"

        return classification_result

    except Exception as e:
        logger.error(f"Error in classification: {e}")
        return {
            "type": "Other",
            "confidence": 0,
            "reasoning": f"Error during classification: {str(e)}"
        }

# -----------------------------------------------------------------------
# 2) CLASSIFY A SINGLE FILE (RETURN CLASSIFICATION + EXTRACTED TEXT)
# -----------------------------------------------------------------------
CLASSIFICATION_PROMPT_SINGLE_FILE = """
You are a classification assistant. The user has provided the text of one document.
Your task:
1) Determine if it's a resume, invoice, contract, job description, or something else.
2) Return JSON with keys: "type", "confidence", and "reasoning".
Example JSON:
{
  "type": "Resume",
  "confidence": 0.95,
  "reasoning": "Mentions candidate experience, education, etc."
}
"""

async def classify_single_file_by_text(file_url: str) -> Dict[str, Any]:
    """
    1. Download file from URL
    2. Extract text
    3. Send that text to LLM to classify
    4. Return classification + the extracted text
       e.g. {
         "classification": {
             "type": "Resume",
             "confidence": 0.95,
             "reasoning": "Candidate info found"
         },
         "extracted_text": "All the text from the file..."
       }
    """

    # 1) Extract text (could be full or partial if the file is huge)
    try:
        full_text = await extract_text_from_url(file_url)
    except Exception as e:
        logger.error(f"Failed to extract text from URL {file_url}: {str(e)}")
        return {
            "classification": {
                "type": "Other",
                "confidence": 0.0,
                "reasoning": f"Could not extract text: {str(e)}"
            },
            "extracted_text": ""
        }

    user_message = f"Document text:\n{full_text}"

    try:
        # 2) Classify the text
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": CLASSIFICATION_PROMPT_SINGLE_FILE},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content
        logger.debug(f"Classification (single file) raw response: {content}")

        # 3) Attempt to parse the JSON
        classification_result = {"type": "Other", "confidence": 0, "reasoning": ""}
        try:
            classification_result = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Could not parse LLM response as JSON.")

        # Ensure structure
        if "type" not in classification_result:
            classification_result["type"] = "Other"
        if "confidence" not in classification_result:
            classification_result["confidence"] = 0
        if "reasoning" not in classification_result:
            classification_result["reasoning"] = "No reasoning"

        return {
            "classification": classification_result,
            "extracted_text": full_text
        }

    except Exception as e:
        logger.error(f"Error in single file classification: {e}")
        return {
            "classification": {
                "type": "Other",
                "confidence": 0,
                "reasoning": f"Error during classification: {str(e)}"
            },
            "extracted_text": ""
        }

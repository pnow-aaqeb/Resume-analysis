# app/utils/text_extraction.py

import httpx
import logging
import tempfile
import os
from unstructured.partition.auto import partition

logger = logging.getLogger(__name__)

async def extract_text_from_url(file_url: str) -> str:
    """
    Download the file from file_url and extract text using unstructured
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(file_url)
            response.raise_for_status()
            file_content = response.content
    except httpx.HTTPError as e:
        logger.error(f"Failed to download file: {e}")
        raise e

    # Write to temp file
    filename = file_url.split("/")[-1]
    suffix = os.path.splitext(filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(file_content)
        temp_file.flush()
        temp_path = temp_file.name

    # Extract text
    try:
        elements = partition(temp_path)
        extracted_text = "\n".join(str(element) for element in elements)
    finally:
        os.unlink(temp_path)

    return extracted_text

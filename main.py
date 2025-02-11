from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
from unstructured.partition.auto import partition
import tempfile
import os
from pydantic import BaseModel
import logging
from typing import Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Extraction Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ExtractionResponse(BaseModel):
    text: str
    status: str
    document_type: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str = "1.0.0"
    service: str = "document-extraction-service"

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify service status
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
    )

async def download_file(url: str) -> bytes:
    """Download file from URL."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.content
    except httpx.HTTPError as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error downloading file: {str(e)}")

@app.post("/extract-from-url", response_model=ExtractionResponse)
async def extract_from_url(file_url: str):
    """
    Extract text from a document using its URL.
    """
    logger.info(f"Received extraction request for URL: {file_url}")
    
    try:
        # Download file
        file_content = await download_file(file_url)
        filename = file_url.split("/")[-1]
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            temp_file.write(file_content)
            temp_file.flush()
            
            # Extract text using unstructured
            elements = partition(temp_file.name)
            extracted_text = "\n".join([str(element) for element in elements])
            
            # Clean up temp file
            os.unlink(temp_file.name)
        logger.info(f"extracted text:{extracted_text}")
        return ExtractionResponse(
            text=extracted_text,
            status="success",
            document_type=os.path.splitext(filename)[1]
        )
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
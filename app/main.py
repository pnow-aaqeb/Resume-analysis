import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import orchestration, resume

app = FastAPI(title="Email & Document Analysis Service")

# Add CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(orchestration.router, prefix="/orchestration", tags=["Orchestration"])
app.include_router(resume.router, prefix="/resume", tags=["Resume"])

@app.get("/health")
async def health_check():
    """
    Basic health check endpoint
    """
    return {"status": "healthy"}

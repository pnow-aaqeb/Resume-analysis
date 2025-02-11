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


# from langchain_community.graphs.age_graph import AGEGraph
# from langchain_experimental.graph_transformers import LLMGraphTransformer
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain_community.llms import Groq
# from langchain_neo4j import Neo4jGraph
# from langchain.docstore.document import Document
# from langchain_openai import ChatOpenAI
# from prisma import Prisma
# from typing import List, Dict
# import asyncio
# from dotenv import load_dotenv
# import os

# load_dotenv()

# class RecruitmentGraphBuilder:
#     def __init__(self, age_config: Dict):
#         neo4j_uri = os.getenv('NEO4J_URI')
#         neo4j_username = os.getenv('NEO4J_USERNAME')
#         neo4j_password = os.getenv('NEO4J_PASSWORD')
        
#         if not all([neo4j_uri, neo4j_username, neo4j_password]):
#             raise ValueError("Missing required Neo4j environment variables. Please check your .env file.")
            
#         self.graph = Neo4jGraph(
#             url=neo4j_uri,
#             username=neo4j_username,
#             password=neo4j_password,
#         )
#         self.llm = ChatOpenAI(
#             api_key="OPENAI_API_KEY",
#             model_name="gpt-40-mini", 
#             temperature=0.0  # Added for consistent outputs
#         )

        
#         self.prisma = Prisma()
        
#         # Initialize text splitter for long emails
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#             separators=["\n\n", "\n", ". ", " ", ""]
#         )
        
#         # Rest of your initialization code remains same...

#     async def process_emails(self, batch_size: int = 100):
#         """Process emails in batches to handle large volume"""
#         await self.prisma.connect()
        
#         try:
#             # Get total count of messages
#             total_messages = await self.prisma.messages.count()
#             print(f"Total messages to process: {total_messages}")
            
#             # Process in batches
#             for skip in range(0, total_messages, batch_size):
#                 messages = await self.prisma.messages.find_many(
#                     skip=skip,
#                     take=batch_size,
#                     select={
#                         'sender_email':True,
#                         'recipients':True,
#                         'subject':True,
#                         'body':True,
#                         'sent_date_time':True,
#                         'received_date_time':True
#                     }
#                 )
                
#                 print(f"Processing batch {skip//batch_size + 1} of {(total_messages + batch_size - 1)//batch_size}")
                
#                 for message in messages:
#                     await self._process_single_message(message)
                
#                 # Optional: Add delay between batches if needed
#                 # await asyncio.sleep(1)
        
#         finally:
#             await self.prisma.disconnect()

#     async def _process_single_message(self, message):
#         """Process a single message from the database"""
        
#         # Extract metadata from message
#         metadata = {
#             'subject': message.subject,
#             'from': message.from_address,  # adjust field names according to your schema
#             'to': message.to_address,      # adjust field names according to your schema
#             'date': message.date,
#             'thread_id': message.thread_id,
#             'message_id': str(message.id)  # keep database ID for reference
#         }
        
#         # Split message content if necessary
#         chunks = self.text_splitter.split_text(message.content)  # adjust field name according to your schema
        
#         # Process each chunk while maintaining email context
#         for i, chunk in enumerate(chunks):
#             graph_data = self.transformer.transform(
#                 Document(
#                     page_content=chunk,
#                     metadata={
#                         **metadata,
#                         'chunk_id': i,
#                         'total_chunks': len(chunks)
#                     }
#                 )
#             )
            
#             # Add to graph with thread linking
#             await self._add_to_graph_with_threading(graph_data, metadata)

#     async def _add_to_graph_with_threading(self, graph_data, metadata):
#         # Add basic entities and relationships
#         self.graph.add_graph_documents(graph_data)
        
#         # Add thread linking - connect emails in the same thread
#         if metadata.get('thread_id'):
#             query = """
#             MATCH (e1:Email {thread_id: $thread_id})
#             MATCH (e2:Email {thread_id: $thread_id})
#             WHERE e1.date < e2.date
#             CREATE (e1)-[:FOLLOWED_BY]->(e2)
#             """
#             self.graph.query(query, {'thread_id': metadata['thread_id']})

# # Usage example
# async def main():
#     age_config = {
#         "host": "103.110.174.29",
#         "port": 5432,
#         "database": "postgres",
#         "user": "postgres.pnats",  # This is your full username from the URL
#         "password": "qpX4XoDrsYUePcca0ucFm56uV2Qj5y3U",
#         "options": "-c search_path=embeddings",  # Specify the schema
# }
    
#     builder = RecruitmentGraphBuilder(age_config)
#     await builder.process_emails(batch_size=100)  # Adjust batch size based on your system's capacity

# if __name__ == "__main__":
#     asyncio.run(main())
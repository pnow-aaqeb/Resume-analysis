from langchain_community.graphs.age_graph import AGEGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_neo4j import Neo4jGraph
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from prisma import Prisma
from typing import List, Dict
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

class RecruitmentGraphBuilder:
    def __init__(self, age_config: Dict):
        neo4j_uri = os.getenv('NEO4J_URI')
        neo4j_username = os.getenv('NEO4J_USERNAME')
        neo4j_password = os.getenv('NEO4J_PASSWORD')
        
        if not all([neo4j_uri, neo4j_username, neo4j_password]):
            raise ValueError("Missing required Neo4j environment variables. Please check your .env file.")
            
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
        )
        
        self.llm = ChatOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            model_name="gpt-4o-mini", 
            temperature=0.0
        )
        
        self.prisma = Prisma()
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        allowed_nodes = ["Person", "Company", "JobDescription", "Candidate", "Project", "Email"]
        allowed_relationships = [
        ("Person", "SENT", "Email"),
        ("Email", "RECEIVED_BY", "Person"),
        ("Person", "WORKS_FOR", "Company"),
        ("Candidate", "APPLIED_TO", "JobDescription")
        ]
        additional_instructions = """
        Focus on extracting recruitment-related information:
        - Identify people involved in recruitment processes
        - Extract company details and job requirements
        - Track candidate interactions and applications
        - Maintain email communication flow
        Only extract information that is explicitly mentioned in the text.
        """

        # Define which properties to extract
        node_properties = ["name", "email", "role", "title", "requirements", "status"]
        relationship_properties = ["date", "status", "priority"]
        self.llm_transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=allowed_nodes,
            # strict_mode=True, 
            node_properties=node_properties,
            relationship_properties=relationship_properties,
            ignore_tool_usage=False, 
            additional_instructions=additional_instructions
        )

    async def process_emails(self, batch_size: int = 100):
        """Process emails in batches to handle large volume"""
        await self.prisma.connect()
        
        try:
            # Get total count of messages
            total_messages = await self.prisma.messages.count()
            print(f"Total messages to process: {total_messages}")
            
            # Process in batches
            for skip in range(0, total_messages, batch_size):
                messages = await self.prisma.messages.find_many(
                    skip=skip,
                    take=batch_size
                )
                
                print(f"Processing batch {skip//batch_size + 1} of {(total_messages + batch_size - 1)//batch_size}")
                
                for message in messages:
                    await self._process_single_message(message)

        finally:
            await self.prisma.disconnect()

    async def _process_single_message(self, message):
        """Process a single message from the database"""
        
        # Extract metadata from message
        metadata = {
            'subject': message.subject,
            'sender_email': message.sender_email,
            'recipients': message.recipients,
            'sent_date': message.sent_date_time,
            'received_date': message.received_date_time,
            'message_id': str(message.id)
        }
          # Split message content if necessary
        chunks = self.text_splitter.split_text(message.body)
    
        # Process each chunk while maintaining email context
        for i, chunk in enumerate(chunks):
            # Create a Document object properly - this is the key fix
            document = Document(
                page_content=chunk,
                metadata={
                    **metadata,
                    'chunk_id': i,
                    'total_chunks': len(chunks)
                }
            )
        
        # Wrap the document in a list as shown in the working example
        graph_data = self.llm_transformer.convert_to_graph_documents([document])
            # Add to graph
        await self._add_to_graph(graph_data, metadata)

    async def _add_to_graph(self, graph_data, metadata):
        """Add data to Neo4j graph"""
        # Add basic entities and relationships
        self.graph.add_graph_documents(graph_data)
        
        # Create query to establish relationships between emails
        query = """
        MATCH (sender:Person {email: $sender_email})
        MATCH (recipient:Person)
        WHERE recipient.email IN $recipients
        CREATE (sender)-[:SENT]->(email:Email {
            id: $message_id,
            subject: $subject,
            sent_date: $sent_date,
            received_date: $received_date
        })-[:RECEIVED_BY]->(recipient)
        """
        
        # Prepare recipients list from JSON
        recipients = metadata['recipients'] if isinstance(metadata['recipients'], list) else []
        
        # Execute query
        self.graph.query(
            query,
            {
                'sender_email': metadata['sender_email'],
                'recipients': recipients,
                'message_id': metadata['message_id'],
                'subject': metadata['subject'],
                'sent_date': metadata['sent_date'],
                'received_date': metadata['received_date']
            }
        )

async def main():
    age_config = {
        "host": "103.110.174.29",
        "port": 5432,
        "database": "postgres",
        "user": "postgres.pnats",
        "password": "qpX4XoDrsYUePcca0ucFm56uV2Qj5y3U",
        "options": "-c search_path=embeddings",
    }
    
    builder = RecruitmentGraphBuilder(age_config)
    await builder.process_emails(batch_size=100)

if __name__ == "__main__":
    asyncio.run(main())
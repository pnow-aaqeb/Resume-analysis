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
        # allowed_nodes = ["Person", "Company", "JobDescription", "Candidate", "Project", "Email"]
        # allowed_relationships = [
        # ("Person", "SENT", "Email"),
        # ("Email", "RECEIVED_BY", "Person"),
        # ("Person", "WORKS_FOR", "Company"),
        # ("Candidate", "APPLIED_TO", "JobDescription")
        # ]
        # additional_instructions = """
        # Focus on extracting recruitment-related information:
        # - Identify people involved in recruitment processes
        # - Extract company details and job requirements
        # - Track candidate interactions and applications
        # - Maintain email communication flow
        # Only extract information that is explicitly mentioned in the text.
        # """
        # allowed_nodes = [
        #     "Person", "Company", "JobDescription", "Candidate", 
        #     "Project", "Email", "Contract", "Invoice", "OfferLetter",
        #     "Prospect", "LeadGeneration", "Opportunity",
        #     "Fulfillment", "Deal", "Sale"
        # ]

        # allowed_relationships = [
        #     # Existing relationships
        #     ("Person", "SENT", "Email"),
        #     ("Email", "RECEIVED_BY", "Person"),
        #     ("Person", "WORKS_FOR", "Company"),
        #     ("Candidate", "APPLIED_TO", "JobDescription"),
            
        #     # Pipeline stage relationships
        #     ("Prospect", "MOVES_TO", "LeadGeneration"),
        #     ("LeadGeneration", "MOVES_TO", "Opportunity"),
        #     ("Opportunity", "MOVES_TO", "Fulfillment"),
        #     ("Fulfillment", "MOVES_TO", "Deal"),
        #     ("Deal", "MOVES_TO", "Sale"),
            
        #     # Stage participation relationships
        #     ("Email", "PART_OF", "Prospect"),
        #     ("Email", "PART_OF", "LeadGeneration"),
        #     ("Contract", "PART_OF", "Deal"),
        #     ("Invoice", "PART_OF", "Sale"),
        #     ("Candidate", "IN_STAGE", "Fulfillment"),
        #     ("Company", "IN_STAGE", "Opportunity")
        # ]

        # additional_instructions = """
        # Extract and connect entities to sales pipeline stages:
        # 1. Prospect Stage: Initial company research emails
        # 2. Lead Generation: Cold emails and first responses
        # 3. Opportunity: Client requirement discussions
        # 4. Fulfillment: Candidate sourcing and interviews
        # 5. Deal: Contract negotiations
        # 6. Sale: Payments and placements

        # For each entity:
        # - Connect emails to their pipeline stage
        # - Link candidates to fulfillment stage
        # - Associate contracts with deal stage
        # - Connect invoices to sale stage
        # - Track company progression through stages
        # """
        allowed_nodes = [
            # Shared nodes (can be connected to multiple companies)
            "ResearchContact",    # Researchers can handle multiple companies
            "Recruiter",         # Recruiters can handle multiple positions
            
            # Unique nodes (part of specific company hierarchy)
            "Company",           # Each company is unique
            "Position",          # Each position belongs to one company
            "Candidate",         # Each candidate application is unique to a position
            "Submission",        # Each submission is unique
            "Interview",         # Each interview is unique
            "Placement",         # Each placement is unique
            
            # Supporting nodes
            "Contract",          # Linked to specific placement
            "Invoice",           # Linked to specific placement
            "Email"             # Can be linked to multiple nodes
        ]

        # Define relationships with their properties
        allowed_relationships = [
            # Core hierarchy (company-specific path)
            ("Position", "BELONGS_TO", "Company"),
            ("Candidate", "APPLIES_TO", "Position"),
            ("Submission", "FOR", "Candidate"),
            ("Interview", "OF", "Submission"),
            ("Placement", "RESULTS_FROM", "Interview"),
            
            # Shared resource relationships
            ("ResearchContact", "HANDLES", "Company"),
            ("Recruiter", "MANAGES", "Position"),
            
            # Supporting relationships
            ("Contract", "FOR", "Placement"),
            ("Invoice", "FOR", "Placement"),
            ("Email", "REFERENCES", "Company"),
            ("Email", "REFERENCES", "Position"),
            ("Email", "REFERENCES", "Candidate")
        ]

        # Define detailed properties for each node type
        node_properties = [
            # Identifier properties (removed 'id' as it's reserved)
            "uuid",             # Use uuid instead of id for unique identifier
            "name",            # Name of entity
            "email",           # Email address
            "phone",           # Phone number
            
            # Business properties
            "company_name",    # For Company nodes
            "position_title",  # For Position nodes
            "requirements",    # For Position nodes
            "status",         # Current status
            "stage",          # Pipeline stage
            
            # Metadata
            "created_at",     # Creation timestamp
            "updated_at",     # Last update timestamp
            "created_by",     # User who created
            "department",     # Owning department
            
            # Hierarchical properties
            "company_uuid",   # Link to parent company
            "position_uuid",  # Link to parent position
            "hierarchy_level" # Level in hierarchy
        ]


        # Define relationship properties as arrays
        relationship_properties = [
            "creation_date",
            "last_updated",
            "created_by",
            "hierarchy_level",
            "association_date",
            "association_type",
            "relation_type",
            "status"
        ]
        # Define relationship properties as arrays
        relationship_properties = [
            "creation_date",
            "last_updated",
            "created_by",
            "hierarchy_level",
            "association_date",
            "association_type",
            "relation_type",
            "status"
        ]
        additional_instructions = """
        Create a hierarchical graph structure following these rules:

        1. Hierarchy Levels (Parent → Child):
           ResearchContact → Company → Position → Candidate → Submission → Interview → Placement

        2. Each level represents a pipeline stage:
           - Level 1 (ResearchContact): Prospect Stage
           - Level 2 (Company): Lead Generation Stage
           - Level 3 (Position): Opportunity Stage
           - Level 4-6 (Candidate/Submission/Interview): Fulfillment Stage
           - Level 7 (Placement): Deal/Sale Stage

        3. Node Creation Rules:
           - Every node must have a single parent (except ResearchContact)
           - Each node must include pipeline_level property
           - Track department ownership at each level
           - Maintain chronological order with creation_date
           - Store full interaction history

        4. Special Rules:
           - Only classify actual clients as Company nodes
           - Never classify ProficientNow as a company
           - Connect all relevant emails to their respective hierarchy levels
           - Track department ownership transitions

        5. Supporting Documents:
           - Attach Contract and Invoice nodes to Placement
           - Link relevant emails to all levels they connect to
        """

        self.llm_transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=allowed_nodes,
            allowed_relationships=allowed_relationships,
            node_properties=node_properties,
            strict_mode=True,
            relationship_properties=relationship_properties,
            additional_instructions=additional_instructions
        )


    async def process_emails(self, batch_size: int = 100):
        """Process emails in batches to handle large volume using transactions"""
        await self.prisma.connect()
        
        try:
            async with self.prisma.tx() as transaction:
                # Get total count of messages within the transaction
                total_messages = await transaction.messages.count()
                print(f"Total messages to process: {total_messages}")
                
                # Process in batches
                for skip in range(0, total_messages, batch_size):
                    messages = await transaction.messages.find_many(
                        skip=skip,
                        take=batch_size,
                        order={
                            'sent_date_time': 'desc'
                        }
                    )
                    
                    print(f"Processing batch {skip//batch_size + 1} of {(total_messages + batch_size - 1)//batch_size}")
                    
                    for message in messages:
                        await self._process_single_message(message)
                        # processed_count += 1
                        # if processed_count % 10 == 0:  # Print progress every 10 messages
                        #     print(f"Processed {processed_count}/{total_messages} messages")
                
                    print(f"Completed batch {current_batch} of {total_batches}")
                        
        except Exception as e:
            print(f"Error processing emails: {e}")
            raise
        finally:
            await self.prisma.disconnect()

    async def _process_single_message(self, message):
        """Process a single message from the database"""
    
        # Prepare metadata as a string
        metadata_str = (
            f"Subject: {message.subject}\n"
            f"Sender: {message.sender_email}\n"
            f"Recipients: {message.recipients}\n"
            f"Sent Date: {message.sent_date_time}\n"
            f"Department:{message.meta_data.get('departments',[])}\n"
            f"Received Date: {message.received_date_time}\n"
            f"Message ID: {message.id}\n"
        )
        
        # Combine metadata with the body text
        full_text = metadata_str + "\n" + message.body
        
        # Split the combined text instead of just the body
        chunks = self.text_splitter.split_text(full_text)
        
        # Create a list of Document objects for each chunk
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                    # You can still include the metadata here if needed for later reference
                    'subject': message.subject,
                    'sender_email': message.sender_email,
                    'recipients': message.recipients,
                    'sent_date': message.sent_date_time,
                    'received_date': message.received_date_time,
                    'departments':message.meta_data.get('departments',[]),
                    'message_id': str(message.id)
                }
            ))
        
        # Convert all document chunks into graph data
        graph_data = self.llm_transformer.convert_to_graph_documents(documents)
        
        # Add to graph
        await self._add_to_graph(graph_data, {
            'subject': message.subject,
            'sender_email': message.sender_email,
            'recipients': message.recipients,
            'sent_date': message.sent_date_time,
            'received_date': message.received_date_time,
            'departments': message.meta_data.get('departments', []),
            'message_id': str(message.id)
        })

    async def _add_to_graph(self, graph_data, metadata):
        """Add processed data to Neo4j with hierarchy enforcement"""
        self.graph.add_graph_documents(graph_data)
        
        # Enforce hierarchical relationships
        hierarchy_query = """
        MATCH (child)
        WHERE child:Company OR child:Position OR child:Candidate 
           OR child:Submission OR child:Interview OR child:Placement
        WITH child
        MATCH (parent)
        WHERE (child:Company AND parent:ResearchContact)
           OR (child:Position AND parent:Company)
           OR (child:Candidate AND parent:Position)
           OR (child:Submission AND parent:Candidate)
           OR (child:Interview AND parent:Submission)
           OR (child:Placement AND parent:Interview)
        MERGE (child)-[r:CHILD_OF]->(parent)
        SET r.creation_date = CASE WHEN r.creation_date IS NULL 
            THEN datetime() ELSE r.creation_date END,
            r.last_updated = datetime(),
            r.hierarchy_level = parent.pipeline_level
        """
        
        # Update hierarchy levels
        level_query = """
        MATCH p=(start:ResearchContact)-[:CHILD_OF*]->(end)
        WITH nodes(p) as nodes
        UNWIND range(0,size(nodes)-1) as i
        WITH nodes[i] as node, i+1 as level
        SET node.pipeline_level = level
        """
        
        # Connect supporting documents
        support_query = """
        MATCH (p:Placement)
        WITH p
        MATCH (doc)
        WHERE (doc:Contract OR doc:Invoice)
        AND NOT (doc)-[:BELONGS_TO]->(p)
        MERGE (doc)-[r:BELONGS_TO]->(p)
        SET r.association_date = datetime()
        """
        
        self.graph.query(hierarchy_query)
        self.graph.query(level_query)
        self.graph.query(support_query)

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
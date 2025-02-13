from langchain_community.graphs.age_graph import AGEGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_neo4j import Neo4jGraph
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from prisma import Prisma
from typing import List, Dict
import asyncio
from dotenv import load_dotenv
import os
import html2text
import json
import logging
from datetime import datetime

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'graph_creation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class RecruitmentGraphBuilder:
    def __init__(self, age_config: Dict):
        neo4j_uri = os.getenv('NEO4J_URI')
        neo4j_username = os.getenv('NEO4J_USERNAME')
        neo4j_password = os.getenv('NEO4J_PASSWORD')
        
        if not all([neo4j_uri, neo4j_username, neo4j_password]):
            raise ValueError("Missing required Neo4j environment variables. Please check your .env file.")
        
        self.html2text=html2text.HTML2Text()
        self.html2text.ignore_links=True
            
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
        
        # Define allowed nodes including shared and unique ones.
        allowed_nodes = [
            # Shared nodes (re-used across company graphs)
            "ResearchContact",    # Researchers initiating outreach
            "BDM",                # Business Development Manager takes over onboarding
            "Recruiter",          # Recruiters handle candidate outreach
            
            # Unique nodes (specific to a company’s recruitment flow)
            "Company",           # The client company (each unique)
            "Position",          # Open position at a company
            "Candidate",         # Candidate applying for a position
            "Submission",        # Processed (and anonymized) resume/submission
            "Interview",         # Interview details
            "Placement",         # Final placement details
            
            # Supporting nodes
            "Contract",          # Contract linked to a placement
            "Invoice",           # Invoice linked to a placement
            "Email"              # Emails referencing any node
        ]

        # Define relationships to reflect the recruitment flow.
        allowed_relationships = [
            # Core hierarchical pipeline (company-specific)
            ("Position", "BELONGS_TO", "Company"),
            ("Candidate", "APPLIES_TO", "Position"),
            ("Submission", "FOR", "Candidate"),
            ("Interview", "OF", "Submission"),
            ("Placement", "RESULTS_FROM", "Interview"),
            
            # Shared resource relationships
            ("ResearchContact", "HANDLES", "Company"),
            ("BDM", "ONBOARDS", "Company"),         # BDM takes over after client agreement
            ("BDM", "ASSIGNS", "Position"),           # BDM assigns positions to recruiters
            ("Recruiter", "MANAGES", "Position"),
            
            # Supporting relationships for documents
            ("Contract", "FOR", "Placement"),
            ("Invoice", "FOR", "Placement"),
            ("Email", "REFERENCES", "Company"),
            ("Email", "REFERENCES", "Position"),
            ("Email", "REFERENCES", "Candidate")
        ]

        # Define properties for nodes
        # node_properties = [
        #     # Identifier properties
        #     "uuid",             # Unique identifier (use uuid instead of id)
        #     "name",             # Name of the entity
        #     "email",            # Email address
        #     "phone",            # Phone number
            
        #     # Business properties
        #     "company_name",     # For Company nodes
        #     "position_title",   # For Position nodes
        #     "requirements",     # For Position nodes
        #     "status",           # Current status in the pipeline
        #     "stage",            # Pipeline stage description
            
        #     # Metadata
        #     "created_at",       # Creation timestamp
        #     "updated_at",       # Last update timestamp
        #     "created_by",       # User who created the node
        #     "department",       # Owning department
            
        #     # Hierarchical properties (to help connect nodes)
        #     "company_uuid",     # Link to parent Company node
        #     "position_uuid",    # Link to parent Position node
        #     "hierarchy_level"   # Level in the recruitment pipeline
        # ]

        # Define properties for relationships
        # relationship_properties = [
        #     "creation_date",
        #     "last_updated",
        #     "created_by",
        #     "hierarchy_level",
        #     "association_date",
        #     "association_type",
        #     "relation_type",
        #     "status"
        # ]

        # In the LLMGraphTransformer setup:
        # additional_instructions = """
        # Create nodes with detailed properties:

        # 1. Company Nodes:
        # MERGE (c:Company {id: company.name})
        # SET c.size = company.size,
        #     c.industry = company.industry,
        #     c.location = company.location,
        #     c.created_at = datetime(),
        #     c.status = company.status

        # 2. Position Nodes:
        # MERGE (p:Position {id: position.title})
        # SET p.company_id = company.id,
        #     p.requirements = position.job_role_description_detailed,
        #     p.salary_range = position.salary_range,
        #     p.location = position.location_city,
        #     p.created_at = datetime(),
        #     p.status = position.status

        # 3. Candidate Nodes:
        # MERGE (can:Candidate {id: candidate.candidate_full_name})
        # SET can.email = candidate.email,
        #     can.phone = candidate.phone,
        #     can.experience = candidate.total_work_experience,
        #     can.location = candidate.location,
        #     can.created_at = datetime(),
        #     can.status = candidate.status

        # 4. Relationships with Properties:
        # MERGE (child)-[r:CHILD_OF]->(parent)
        # SET r.creation_date = datetime(),
        #     r.last_updated = datetime(),
        #     r.status = 'ACTIVE'

        # Only create nodes for validated entities and include all available properties from the database context.
        # """
        additional_instructions = """
        Create a hierarchical graph structure using ONLY the validated entities provided in the metadata:

        1. Company Nodes:
        - Create Company nodes ONLY for companies listed in metadata.validated_entities.companies
        - Use exact company names as provided

        2. Position Nodes:
        - Create Position nodes ONLY for positions listed in metadata.validated_entities.positions
        - Ensure positions are linked to their correct companies

        3. Candidate Nodes:
        - Create Candidate nodes for names in metadata.validated_entities.candidates
        - Link candidates to appropriate positions based on context

        4. Hierarchy Rules:
        ResearchContact → Company → Position → Candidate → Submission → Interview → Placement

        5. Additional Rules:
        - Only use validated entity names exactly as provided
        - Each node must include pipeline_level property
        - Track department ownership and timestamps
        - Link supporting documents appropriately

        Strictly use only the validated entities from metadata.validated_entities for creating nodes.
        """

        self.llm_transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=allowed_nodes,
            allowed_relationships=allowed_relationships,
            # node_properties=node_properties,
            # strict_mode=True,
            # relationship_properties=relationship_properties,
            additional_instructions=additional_instructions
        )


    async def process_emails(self, batch_size: int = 100):
        """Process emails in batches using transactions to handle large volumes."""
        await self.prisma.connect()
        
        try:
            async with self.prisma.tx() as transaction:
                total_messages = await transaction.messages.count()
                print(f"Total messages to process: {total_messages}")
                
                total_batches = (total_messages + batch_size - 1) // batch_size
                for skip in range(0, total_messages, batch_size):
                    messages = await transaction.messages.find_many(
                        skip=skip,
                        take=batch_size,
                        order={'sent_date_time': 'desc'}
                    )
                    
                    current_batch = skip // batch_size + 1
                    print(f"Processing batch {current_batch} of {total_batches}")
                    
                    for message in messages:
                        await self._process_single_message(message)
                
                    print(f"Completed batch {current_batch} of {total_batches}")
                        
        except Exception as e:
            print(f"Error processing emails: {e}")
            raise
        finally:
            await self.prisma.disconnect()

    async def _extract_entities(self, email_content: str):
        """Extract potential entities from email content using LLM"""
        extraction_prompt = f"""
        Your task is to extract entities from the email content and return them in a specific JSON format.
        You must return ONLY valid JSON, no additional text or explanations.
        
        Return this exact format, even if no entities are found:
        {{
            "companies": [],
            "positions": [],
            "candidates": []
        }}

        Rules:
        1. Exclude ProficientNow from companies
        2. Only include actual client companies
        3. Extract position titles and job roles
        4. Extract candidate names if mentioned
        
        Email content:
        {email_content}
        """
        
        messages = [HumanMessage(content=extraction_prompt)]
        try:
            response = await self.llm.ainvoke(messages)
            logging.debug(f"LLM Response: {response.content}")
            
            # Try to clean the response if it's not pure JSON
            content = response.content.strip()
            # Find the first { and last }
            start = content.find('{')
            end = content.rfind('}')
            
            if start != -1 and end != -1:
                content = content[start:end+1]
                
            try:
                extracted = json.loads(content)
                # Validate the structure
                if not all(key in extracted for key in ['companies', 'positions', 'candidates']):
                    logging.warning(f"Missing required keys in extracted data: {extracted}")
                    return {"companies": [], "positions": [], "candidates": []}
                    
                logging.info(f"Successfully extracted entities: {json.dumps(extracted, indent=2)}")
                return extracted
                
            except json.JSONDecodeError as je:
                logging.error(f"JSON Decode Error: {je}")
                logging.error(f"Attempted to parse: {content}")
                return {"companies": [], "positions": [], "candidates": []}
                
        except Exception as e:
            logging.error(f"Error in entity extraction: {str(e)}")
            logging.error(f"Email content length: {len(email_content)}")
            return {"companies": [], "positions": [], "candidates": []}

    async def _fetch_company_context(self, email_content: str):
        """Fetch relevant company data based on email content"""
        logging.info("Fetching company context...")
        extracted = await self._extract_entities(email_content)
        
        if not extracted['companies']:
            logging.info("No companies extracted from email")
            return []
        
        try:
            logging.info(f"Searching for companies: {extracted['companies']}")
            companies = await self.prisma.companies.find_many(
                where={
                    'OR': [
                        {'name': {'contains': company_name}} 
                        for company_name in extracted['companies']
                    ]
                },
                include={
                    'positions': True,
                    'contacts': True,
                    'leads': True
                }
            )
            logging.info(f"Found {len(companies)} matching companies")
            return companies
        except Exception as e:
            logging.error(f"Error fetching company data: {str(e)}")
            return []

    async def _fetch_position_context(self, email_content: str):
        """Fetch relevant position data"""
        logging.info("Fetching position context...")
        extracted = await self._extract_entities(email_content)
        
        if not extracted['positions']:
            logging.info("No positions extracted from email")
            return []
            
        try:
            logging.info(f"Searching for positions: {extracted['positions']}")
            positions = await self.prisma.positions.find_many(
                where={
                    'OR': [
                        {'title': {'contains': position_title}}
                        for position_title in extracted['positions']
                    ]
                },
                include={
                    'job_roles': True,
                    'pocs': True,
                    'documents': True
                }
            )
            logging.info(f"Found {len(positions)} matching positions")
            return positions
        except Exception as e:
            logging.error(f"Error fetching position data: {str(e)}")
            return []

    async def _fetch_candidate_context(self, email_content: str):
        """Fetch relevant candidate data"""
        logging.info("Fetching candidate context...")
        extracted = await self._extract_entities(email_content)
        
        if not extracted['candidates']:
            logging.info("No candidates found")
            return []
        try:
            logging.info(f"Searching for candidate data: {extracted['candidates']}")
            candidates = await self.prisma.candidates.find_many(
                where={
                    'OR': [
                        {'candidate_full_name': {'contains': name}}
                        for name in extracted['candidates']
                    ]
                },
                include={
                    'work_experiences': True,
                    'education': True,
                    'certifications': True,
                    'candidate_submissions': True
                }
            )
        except Exception as e:
            logging.error(f"Error fetching candidate data:{str(e)}")
        return candidates

    async def _extract_valid_entities(self, email_content: str):
        """Extract and validate all entities against database"""
        extracted = await self._extract_entities(email_content)
        valid_data = {
            "companies": [],
            "positions": [],
            "candidates": [],
            "exact_matches": {  # Store exact matches to use in graph
                "companies": {},  # Will store {extracted_name: database_name}
                "positions": {},
                "candidates": {}
            }
        }

        # Validate companies
        if extracted['companies']:
            companies = await self.prisma.companies.find_many(
                where={
                    'OR': [
                        {'name': {'contains': company_name}} 
                        for company_name in extracted['companies']
                    ]
                }
            )
            # Store exact database names
            for company in companies:
                matched_name = next((name for name in extracted['companies'] 
                                if name.lower() in company.name.lower()), None)
                if matched_name:
                    valid_data['exact_matches']['companies'][matched_name] = company.name
                    valid_data['companies'].append(company.name)

            logging.info(f"Companies validation - Original: {extracted['companies']}, Valid: {valid_data['companies']}")
            
        # Validate positions
        if extracted['positions']:
            positions = await self.prisma.positions.find_many(
                where={
                    'OR': [
                        {'title': {'contains': position_title}}
                        for position_title in extracted['positions']
                    ]
                }
            )
            # Store exact database titles
            for position in positions:
                matched_title = next((title for title in extracted['positions'] 
                                    if title.lower() in position.title.lower()), None)
                if matched_title:
                    valid_data['exact_matches']['positions'][matched_title] = position.title
                    valid_data['positions'].append(position.title)

            logging.info(f"Positions validation - Original: {extracted['positions']}, Valid: {valid_data['positions']}")

        # Validate candidates
        if extracted['candidates']:
            candidates = await self.prisma.candidates.find_many(
                where={
                    'OR': [
                        {'candidate_full_name': {'contains': name}}
                        for name in extracted['candidates']
                    ]
                }
            )
            # Store exact database names
            for candidate in candidates:
                matched_name = next((name for name in extracted['candidates'] 
                                if name.lower() in candidate.candidate_full_name.lower()), None)
                if matched_name:
                    valid_data['exact_matches']['candidates'][matched_name] = candidate.candidate_full_name
                    valid_data['candidates'].append(candidate.candidate_full_name)

            logging.info(f"Candidates validation - Original: {extracted['candidates']}, Valid: {valid_data['candidates']}")

        # Update instructions with exact matches
        additional_context = f"""
        Use these exact names from the database:
        Companies: {json.dumps(valid_data['exact_matches']['companies'], indent=2)}
        Positions: {json.dumps(valid_data['exact_matches']['positions'], indent=2)}
        Candidates: {json.dumps(valid_data['exact_matches']['candidates'], indent=2)}
        
        Important: Only use the exact names listed above when creating nodes.
        """
    
        return valid_data, additional_context

    def _convert_to_dict(self, obj):
        """Convert Prisma models to dictionaries"""
        if hasattr(obj, '__dict__'):
            return {key: self._convert_to_dict(value) 
                    for key, value in obj.__dict__.items() 
                    if not key.startswith('_')}
        elif isinstance(obj, list):
            return [self._convert_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._convert_to_dict(value) 
                    for key, value in obj.items()}
        else:
            try:
                # Try to convert to JSON serializable format
                json.dumps(obj)
                return obj
            except (TypeError, OverflowError):
                # If object can't be serialized, convert to string
                return str(obj)

    async def _process_single_message(self, message):
        """Process a single message from the database."""
        try:
            logging.info(f"\n{'='*50}\nProcessing Message ID: {message.id}\n{'='*50}")
            
            # Extract email content
            email_text = self.html2text.handle(message.body)
            
            # Extract entities once
            extracted_entities = await self._extract_entities(email_text)
            logging.info(f"Initially extracted entities: {json.dumps(extracted_entities, indent=2)}")
            
            # Early return if no companies found
            if not extracted_entities['companies']:
                logging.info(f"No companies found in message {message.id}")
                return

            # Validate companies exist in database
            companies = await self.prisma.companies.find_many(
                where={
                    'OR': [
                        {'name': {'contains': company_name}} 
                        for company_name in extracted_entities['companies']
                    ]
                },
                include={
                    'positions': True,
                    'contacts': True
                }
            )

            if not companies:
                logging.info(f"No matching companies found in database for: {extracted_entities['companies']}")
                return

            # Only proceed if we have valid companies
            valid_company_names = [company.name for company in companies]
            logging.info(f"Found valid companies: {valid_company_names}")

            # Prepare metadata
            metadata = {
                'subject': message.subject,
                'sender_email': message.sender_email,
                'recipients': message.recipients,
                'sent_date': message.sent_date_time,
                'received_date': message.received_date_time,
                'departments': message.meta_data.get('departments', []),
                'message_id': str(message.id),
                'entities': {
                    'companies': valid_company_names,
                    'positions': extracted_entities['positions'],
                    'candidates': extracted_entities['candidates']
                }
            }

            # Create document with enriched context
            context_str = (
                f"Subject: {message.subject}\n"
                f"Sender: {message.sender_email}\n"
                f"Recipients: {message.recipients}\n"
                f"Sent Date: {message.sent_date_time}\n"
                f"Companies: {valid_company_names}\n"
                f"Positions: {extracted_entities['positions']}\n"
                f"Candidates: {extracted_entities['candidates']}\n\n"
                f"Email Content:\n{email_text}"
            )

            # Split into chunks
            chunks = self.text_splitter.split_text(context_str)
            documents = [
                Document(
                    page_content=chunk,
                    metadata=metadata
                ) for i, chunk in enumerate(chunks)
            ]

            # Convert to graph data
            logging.info("Converting to graph data...")
            graph_data = self.llm_transformer.convert_to_graph_documents(documents)
            
            if graph_data:
                logging.info("Creating graph nodes and relationships:")
                for item in graph_data:
                    if 'nodes' in item:
                        for node in item['nodes']:
                            logging.info(f"Node: {node.get('type')} - {node.get('properties', {})}")
                    if 'relationships' in item:
                        for rel in item['relationships']:
                            logging.info(f"Relationship: {rel.get('source')} -> {rel.get('type')} -> {rel.get('target')}")
                
                # Add to graph
                await self._add_to_graph(graph_data, metadata)
                logging.info(f"Successfully processed message {message.id}")
            else:
                logging.info(f"No graph data generated for message {message.id}")

        except Exception as e:
            logging.error(f"Error processing message {message.id}: {str(e)}", exc_info=True)
            """Process a single message from the database."""
            # Prepare metadata string for context.
            metadata_str = (
                f"Subject: {message.subject}\n"
                f"Sender: {message.sender_email}\n"
                f"Recipients: {message.recipients}\n"
                f"Sent Date: {message.sent_date_time}\n"
                f"Department: {message.meta_data.get('departments', [])}\n"
                f"Received Date: {message.received_date_time}\n"
                f"Message ID: {message.id}\n"
            )

            # Extract potential entities from email
            email_text = self.html2text.handle(message.body)
            valid_data, additional_context = await self._extract_valid_entities(email_text)
            
            if not valid_data['companies']:
                logging.info(f"Skipping message {message.id} - No valid companies found")
                return
            
            # Fetch relevant context from database
            company_context = await self._fetch_company_context(email_text)
            position_context = await self._fetch_position_context(email_text)
            candidate_context = await self._fetch_candidate_context(email_text)
            
            # Add context to metadata
            if company_context:
                metadata_str += "\nCompany Context:\n" + json.dumps(self._convert_to_dict(company_context), indent=2)
            if position_context:
                metadata_str += "\nPosition Context:\n" + json.dumps(self._convert_to_dict(position_context), indent=2)
            if candidate_context:
                metadata_str += "\nCandidate Context:\n" + json.dumps(self._convert_to_dict(candidate_context), indent=2)
            
            metadata_str += "\nValidated Entities:\n" + json.dumps(valid_data, indent=2)
            
            # Combine all text
            full_text = metadata_str + "\n" + email_text
            
            # Split text into chunks.
            chunks = self.text_splitter.split_text(full_text)
            
            # Create Document objects for each chunk.
            documents = [
                Document(
                    page_content=chunk,
                    metadata={
                       'validated_entities': valid_data['exact_matches'],
                        'message_id': str(message.id),
                        'sender': message.sender_email,
                        'sent_date': message.sent_date_time,
                        'departments': message.meta_data.get('departments', [])
                    }
                ) for i, chunk in enumerate(chunks)
            ]
            
            # Convert document chunks into structured graph data.
            graph_data = self.llm_transformer.convert_to_graph_documents(documents)
            
            
            # Add the extracted graph data to Neo4j.
            await self._add_to_graph(graph_data, {
                'subject': message.subject,
                'sender_email': message.sender_email,
                'recipients': message.recipients,
                'sent_date': message.sent_date_time,
                'received_date': message.received_date_time,
                'departments': message.meta_data.get('departments', []),
                'message_id': str(message.id)
            })

    # async def _add_to_graph(self, graph_data, metadata):
    #     """Add processed graph data to Neo4j with detailed properties."""
    #     # First add the basic graph data
    #     self.graph.add_graph_documents(graph_data)
        
    #     # Update company properties
    #     company_query = """
    #     MERGE (c:Company {uuid: $company_uuid})
    #     SET c.name = $company_name,
    #         c.industry = $industry,
    #         c.location = $location,
    #         c.created_at = datetime($created_timestamp),
    #         c.status = $company_status,
    #         c.pipeline_level = 'Client Acquisition'
    #     WITH c
    #     MERGE (rc:ResearchContact {email: $research_contact_email})
    #     SET rc.name = $research_contact_name,
    #         rc.department = 'Research'
    #     MERGE (rc)-[r:HANDLES {since: date()}]->(c)
    #     SET r.association_type = 'Client Outreach'

    #     3. Candidate Nodes:
    #     MERGE (can:Candidate {id: candidate.candidate_full_name})
    #     SET can.email = candidate.email,
    #         can.phone = candidate.phone,
    #         can.experience = candidate.total_work_experience,
    #         can.location = candidate.location,
    #         can.created_at = datetime(),
    #         can.status = candidate.status

    #     4. Relationships with Properties:
    #     MERGE (child)-[r:CHILD_OF]->(parent)
    #     SET r.creation_date = datetime(),
    #         r.last_updated = datetime(),
    #         r.status = 'ACTIVE'
    #     """
        
    #     # Update position properties
    #     position_query = """
    #      2. Position Nodes:
    #     MERGE (p:Position {id: position.title})
    #     SET p.company_id = company.id,
    #         p.requirements = position.job_role_description_detailed,
    #         p.salary_range = position.salary_range,
    #         p.location = position.location_city,
    #         p.created_at = datetime(),
    #         p.status = position.status
    #     """
        
    #     # Update candidate properties
    #     candidate_query = """
    #      3. Candidate Nodes:
    #     MERGE (can:Candidate {id: candidate.candidate_full_name})
    #     SET can.email = candidate.email,
    #         can.phone = candidate.phone,
    #         can.experience = candidate.total_work_experience,
    #         can.location = candidate.location,
    #         can.created_at = datetime(),
    #         can.status = candidate.status
    #     """
        
    #     # Execute the property updates
    #     if metadata.get('companies'):
    #         self.graph.query(company_query, {
    #             'company_names': [c.name for c in metadata['companies']],
    #             'company_data': [self._convert_to_dict(c) for c in metadata['companies']]
    #         })
        
    #     if metadata.get('positions'):
    #         self.graph.query(position_query, {
    #             'position_titles': [p.title for p in metadata['positions']],
    #             'position_data': [self._convert_to_dict(p) for p in metadata['positions']]
    #         })
        
    #     if metadata.get('candidates'):
    #         self.graph.query(candidate_query, {
    #             'candidate_names': [c.candidate_full_name for c in metadata['candidates']],
    #             'candidate_data': [self._convert_to_dict(c) for c in metadata['candidates']]
    #         })
    async def _add_to_graph(self, graph_data, metadata):
        """Add recruitment data to Neo4j with a simple, clear structure."""
        
        recruitment_query = """
        MERGE (c:Company {name: company.id})
        SET c.industry = $industry,
            c.location = $location,
            c.status = $status
        
        WITH c
        UNWIND $positions AS position
        MERGE (p:Position {title: position.id})
        SET p.requirements = position.requirements,
            p.status = position.status
        MERGE (p)-[:BELONGS_TO]->(c)
        
        WITH p
        UNWIND $candidates AS candidate
        MERGE (can:Candidate {name: candidate.id})
        SET can.email = candidate.email,
            can.status = candidate.status
        MERGE (can)-[:APPLIES_TO]->(p)
        """
        
        try:
            # params = {
            #     'company_name': metadata['company_name'],
            #     'industry': metadata.get('industry', ''),
            #     'location': metadata.get('location', ''),
            #     'status': metadata.get('status', 'Active'),
            #     'positions': [{
            #         'title': pos.get('title', ''),
            #         'requirements': pos.get('requirements', ''),
            #         'status': pos.get('status', 'Open')
            #     } for pos in metadata.get('positions', [])],
            #     'candidates': [{
            #         'name': cand.get('name', ''),
            #         'email': cand.get('email', ''),
            #         'status': cand.get('status', 'Active')
            #     } for cand in metadata.get('candidates', [])]
            # }
            
            self.graph.query(recruitment_query)
            logging.info(f"Successfully added data for company {metadata['company_name']}")
            
        except Exception as e:
            logging.error(f"Error adding to graph: {str(e)}")
            raise

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
    async def _add_to_graph(self, graph_data, metadata):
        """Add processed graph data to Neo4j with detailed properties."""
        # First add the basic graph data
        self.graph.add_graph_documents(graph_data)
        
        # Update company properties
        company_query = """
        MATCH (c:Company)
        WHERE c.id IN $company_names
        WITH c, $company_data as data
        WHERE c.id = data.name
        SET c.size = data.size,
            c.industry = data.industry,
            c.location = data.location,
            c.created_at = datetime(),
            c.status = data.status
        """
        
        # Update position properties
        position_query = """
        MATCH (p:Position)
        WHERE p.id IN $position_titles
        WITH p, $position_data as data
        WHERE p.id = data.title
        SET p.requirements = data.job_role_description_detailed,
            p.salary_range = data.salary_range,
            p.location = data.location_city,
            p.created_at = datetime(),
            p.status = data.status
        """
        
        # Update candidate properties
        candidate_query = """
        MATCH (can:Candidate)
        WHERE can.id IN $candidate_names
        WITH can, $candidate_data as data
        WHERE can.id = data.candidate_full_name
        SET can.email = data.email,
            can.phone = data.phone,
            can.experience = data.total_work_experience,
            can.location = data.location,
            can.created_at = datetime(),
            can.status = data.status
        """
        
        # Execute the property updates
        if metadata.get('companies'):
            self.graph.query(company_query, {
                'company_names': [c.name for c in metadata['companies']],
                'company_data': [self._convert_to_dict(c) for c in metadata['companies']]
            })
        
        if metadata.get('positions'):
            self.graph.query(position_query, {
                'position_titles': [p.title for p in metadata['positions']],
                'position_data': [self._convert_to_dict(p) for p in metadata['positions']]
            })
        
        if metadata.get('candidates'):
            self.graph.query(candidate_query, {
                'candidate_names': [c.candidate_full_name for c in metadata['candidates']],
                'candidate_data': [self._convert_to_dict(c) for c in metadata['candidates']]
            })
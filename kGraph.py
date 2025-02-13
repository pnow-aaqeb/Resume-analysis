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
        additional_instructions = """
        Create a hierarchical graph structure using ONLY validated entities:

        1. Company Nodes:
        - Create ONLY for companies in metadata.validated_entities.companies
        - The id and name MUST be the exact company name from metadata.validated_entities.companies
        - DO NOT create generic company nodes (like "Company A")

        2. Position Nodes:
        - Create ONLY for positions in metadata.validated_entities.positions
        - The id and title MUST be the exact position title from metadata.validated_entities.positions
        - DO NOT create positions without a validated company

        3. Candidate Nodes:
        - Create ONLY for candidates in metadata.validated_entities.candidates
        - The id and name MUST be the exact candidate name from metadata.validated_entities.candidates
        - DO NOT create generic candidate nodes

        4. Hierarchy Rules:
        - Only create child nodes if parent exists
        - ResearchContact → Company → Position → Candidate → Submission → Interview → Placement
        - Each node must belong to valid parent in hierarchy

        5. Additional Rules:
        - Use only validated entity names exactly as provided
        - Do not create nodes for unvalidated entities
        - Maintain proper hierarchy with CHILD_OF relationships
        - Track creation timestamps and department ownership


        IMPORTANT RULES:
        - ONLY use names exactly as they appear in metadata.validated_entities
        - DO NOT create any nodes with generic names like "Company A" or "Candidate 1"
        - Each node MUST have an id that matches its exact name from validated_entities
        - If an entity is not in metadata.validated_entities, DO NOT create a node for it
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

    # async def _extract_valid_entities(self, email_content: str):
    #     """Extract and validate all entities against database"""
    #     try:
    #         # Get initial extracted entities
    #         extracted = await self._extract_entities(email_content)
            
    #         # Initialize valid_data structure
    #         valid_data = {
    #             "companies": [],
    #             "positions": [],
    #             "candidates": [],
    #             "exact_matches": {
    #                 "companies": {},
    #                 "positions": {},
    #                 "candidates": {}
    #             }
    #         }

    #         # Validate companies
    #         if extracted['companies']:
    #             companies = await self.prisma.companies.find_many(
    #                 where={
    #                     'OR': [
    #                         {'name': {'contains': company_name}} 
    #                         for company_name in extracted['companies']
    #                     ],
    #                     'is_deleted': False
    #                 }
    #             )
                
    #             for company in companies:
    #                 matched_name = next(
    #                     (name for name in extracted['companies'] 
    #                     if name.lower() in company.name.lower()),
    #                     None
    #                 )
    #                 if matched_name:
    #                     valid_data['exact_matches']['companies'][matched_name] = company.name
    #                     if company.name not in valid_data['companies']:
    #                         valid_data['companies'].append(company.name)

    #             logging.info(f"Companies validation - Original: {extracted['companies']}, Valid: {valid_data['companies']}")
                
    #         # Validate positions
    #         if extracted['positions']:
    #             positions = await self.prisma.positions.find_many(
    #                 where={
    #                     'OR': [
    #                         {'title': {'contains': position_title}}
    #                         for position_title in extracted['positions']
    #                     ],
    #                     'is_deleted': False,
    #                     'is_company_deleted': False
    #                 },
    #                 include={
    #                     'companies': True
    #                 }
    #             )
                
    #             for position in positions:
    #                 matched_title = next(
    #                     (title for title in extracted['positions'] 
    #                     if title.lower() in position.title.lower()),
    #                     None
    #                 )
    #                 if matched_title:
    #                     valid_data['exact_matches']['positions'][matched_title] = position.title
    #                     if position.title not in valid_data['positions']:
    #                         valid_data['positions'].append(position.title)
                        
    #                     # Add associated company if not already included
    #                     if position.company_name and position.company_name not in valid_data['companies']:
    #                         valid_data['companies'].append(position.company_name)
    #                         valid_data['exact_matches']['companies'][position.company_name] = position.company_name

    #             logging.info(f"Positions validation - Original: {extracted['positions']}, Valid: {valid_data['positions']}")

    #         # Validate candidates (simplified)
    #         if extracted['candidates']:
    #             candidates = await self.prisma.candidates.find_many(
    #                 where={
    #                     'OR': [
    #                         {'candidate_full_name': {'contains': name}}
    #                         for name in extracted['candidates']
    #                     ]
    #                 }
    #             )
                
    #             for candidate in candidates:
    #                 matched_name = next(
    #                     (name for name in extracted['candidates'] 
    #                     if name.lower() in candidate.candidate_full_name.lower()),
    #                     None
    #                 )
    #                 if matched_name:
    #                     valid_data['exact_matches']['candidates'][matched_name] = candidate.candidate_full_name
    #                     if candidate.candidate_full_name not in valid_data['candidates']:
    #                         valid_data['candidates'].append(candidate.candidate_full_name)

    #             logging.info(f"Candidates validation - Original: {extracted['candidates']}, Valid: {valid_data['candidates']}")

    #         # Update instructions with exact matches
    #         additional_context = f"""
    #         Use these exact names from the database:
    #         Companies: {json.dumps(valid_data['exact_matches']['companies'], indent=2)}
    #         Positions: {json.dumps(valid_data['exact_matches']['positions'], indent=2)}
    #         Candidates: {json.dumps(valid_data['exact_matches']['candidates'], indent=2)}
            
    #         Important: Only use the exact names listed above when creating nodes.
    #         """
            
    #         return valid_data, additional_context
            
    #     except Exception as e:
    #         logging.error(f"Error in entity validation: {str(e)}")
    #         # Return empty but valid structure
    #         return {
    #             "companies": [],
    #             "positions": [],
    #             "candidates": [],
    #             "exact_matches": {"companies": {}, "positions": {}, "candidates": {}}
    #         }, ""
    async def _check_user_name(self, name: str):
        """Simple check if name matches any user's real or pseudo name"""
        try:
            name = name.strip()
            
            # Split into first and last name if possible
            names = name.split()
            first_name = names[0] if names else ""
            last_name = names[-1] if len(names) > 1 else ""
            
            # Simple query to check real name
            user = await self.prisma.users.find_first(
                where={
                    'first_name': first_name,
                    'last_name': last_name
                }
            )
            
            if user:
                return True
                
            # Simple query to check pseudo name
            pseudo_user = await self.prisma.users.find_first(
                where={
                    'psuedo_first_name': first_name,
                    'psuedo_last_name': last_name
                }
            )
            
            return pseudo_user is not None
            
        except Exception as e:
            logging.error(f"Error checking user name: {str(e)}")
            return False
    async def _extract_valid_entities(self, email_content: str):
        """Extract and validate all entities against database"""
        try:
            # Get initial extracted entities
            extracted = await self._extract_entities(email_content)
            
            # Initialize valid_data structure
            valid_data = {
                "companies": [],
                "positions": [],
                "candidates": [],
                "exact_matches": {
                    "companies": {},
                    "positions": {},
                    "candidates": {}
                }
            }

            # Validate companies
            if extracted['companies']:
                companies = await self.prisma.companies.find_many(
                    where={
                        'OR': [
                            {'name': {'equals': company_name.strip()}} 
                            for company_name in extracted['companies']
                        ],
                        'is_deleted': False
                    }
                )
                
                for company in companies:
                    for extracted_name in extracted['companies']:
                        if extracted_name.strip() == company.name.strip():
                            valid_data['exact_matches']['companies'][extracted_name] = company.name
                            if company.name not in valid_data['companies']:
                                valid_data['companies'].append(company.name)
                                break

                logging.info(f"Companies validation - Original: {extracted['companies']}, Valid: {valid_data['companies']}")
                
            # Validate positions with company context
            if extracted['positions']:
                positions = await self.prisma.positions.find_many(
                    where={
                        'OR': [
                            {'title': {'equals': position_title.strip()}}
                            for position_title in extracted['positions']
                        ],
                        'is_deleted': False,
                        'is_company_deleted': False
                    },
                    include={
                        'companies': True
                    }
                )
                
                for position in positions:
                    for extracted_title in extracted['positions']:
                        if extracted_title.strip() == position.title.strip():
                            valid_data['exact_matches']['positions'][extracted_title] = position.title
                            if position.title not in valid_data['positions']:
                                valid_data['positions'].append(position.title)
                            
                                # Add associated company if valid and not already included
                                if position.company_name and not position.is_company_deleted:
                                    if position.company_name not in valid_data['companies']:
                                        valid_data['companies'].append(position.company_name)
                                        valid_data['exact_matches']['companies'][position.company_name] = position.company_name
                                break

                logging.info(f"Positions validation - Original: {extracted['positions']}, Valid: {valid_data['positions']}")

            # Validate candidates
            if extracted['candidates']:
                # First check candidates table
                candidates = await self.prisma.candidates.find_many(
                    where={
                        'OR': [
                            {'candidate_full_name': {'equals': name.strip()}}
                            for name in extracted['candidates']
                        ]
                    }
                )
                
                # Track which names we've already validated
                validated_names = set()
                
                # First process candidates from database
                for candidate in candidates:
                    for extracted_name in extracted['candidates']:
                        if extracted_name.strip() == candidate.candidate_full_name.strip():
                            valid_data['exact_matches']['candidates'][extracted_name] = candidate.candidate_full_name
                            if candidate.candidate_full_name not in valid_data['candidates']:
                                valid_data['candidates'].append(candidate.candidate_full_name)
                                validated_names.add(extracted_name)
                                break
                
                # For remaining unvalidated candidates, check if they're users
                for extracted_name in extracted['candidates']:
                    if extracted_name not in validated_names:
                        # Only add if NOT a user
                        is_user = await self._check_user_name(extracted_name)
                        if not is_user:
                            # If not in candidates table and not a user, add to valid data
                            if extracted_name not in valid_data['candidates']:
                                valid_data['candidates'].append(extracted_name)
                                valid_data['exact_matches']['candidates'][extracted_name] = extracted_name

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
            
        except Exception as e:
            logging.error(f"Error in entity validation: {str(e)}")
            return {
                "companies": [],
                "positions": [],
                "candidates": [],
                "exact_matches": {"companies": {}, "positions": {}, "candidates": {}}
            }, ""

    async def _process_single_message(self, message):
        """Process a single message from the database."""
        try:
            logging.info(f"\n{'='*50}\nProcessing Message ID: {message.id}\n{'='*50}")
            
            # Extract email content
            email_text = self.html2text.handle(message.body)
            
            # Get validated entities first
            valid_data, additional_context = await self._extract_valid_entities(email_text)
            
            # Early return if no companies found
            if not valid_data['companies']:
                logging.info(f"No companies found in message {message.id}")
                return

            # Create metadata with validated entities
            metadata = {
                'subject': message.subject,
                'sender_email': message.sender_email,
                'recipients': message.recipients,
                'sent_date': message.sent_date_time,
                'received_date': message.received_date_time,
                'departments': message.meta_data.get('departments', []),
                'message_id': str(message.id),
                'validated_entities': valid_data['exact_matches']  # Now valid_data is defined
            }

            # Prepare context string
            context_str = (
                f"Subject: {message.subject}\n"
                f"Sender: {message.sender_email}\n"
                f"Recipients: {message.recipients}\n"
                f"Sent Date: {message.sent_date_time}\n"
                f"Companies: {valid_data['companies']}\n"
                f"Positions: {valid_data['positions']}\n"
                f"Candidates: {valid_data['candidates']}\n\n"
                f"Email Content:\n{email_text}"
            )

            # Split into chunks
            chunks = self.text_splitter.split_text(context_str)
            # In _process_single_message:
            documents = [
                Document(
                    page_content=chunk,
                    metadata={
                        'chunk_id': i,
                        'total_chunks': len(chunks),
                        'subject': message.subject,
                        'sender_email': message.sender_email,
                        'recipients': message.recipients,
                        'sent_date': message.sent_date_time,
                        'departments': message.meta_data.get('departments', []),
                        'message_id': str(message.id),
                        'validated_entities': {
                            'companies': valid_data['companies'],  # List of validated company names
                            'positions': valid_data['positions'],  # List of validated position titles
                            'candidates': valid_data['candidates'],  # List of validated candidate names
                            'exact_matches': valid_data['exact_matches']  # Original to exact mapping
                        },
                        'additional_instructions': additional_context
                    }
                ) for i, chunk in enumerate(chunks)
            ]

            # Convert to graph data
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
            # Don't duplicate processing code here - just log the error
    async def _add_to_graph(self, graph_data, metadata):
        """Add processed graph data to Neo4j and enforce hierarchy."""
        self.graph.add_graph_documents(graph_data)
        
        # Enforce parent-child hierarchy for the core recruitment flow.
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
        SET r.creation_date = COALESCE(r.creation_date, datetime()),
            r.last_updated = datetime(),
            r.hierarchy_level = parent.pipeline_level
        """
        # Replace the hierarchy_query section with:
        # hierarchy_query = """
        # MATCH (child)
        # WHERE child:Company OR child:Position OR child:Candidate 
        #     OR child:Submission OR child:Interview OR child:Placement
        # WITH child
        # MATCH (parent)
        # WHERE (parent:ResearchContact AND child:Company)  # Changed direction
        #     OR (child:Position AND parent:Company)
        #     OR (child:Candidate AND parent:Position)
        #     OR (child:Submission AND parent:Candidate)
        #     OR (child:Interview AND parent:Submission)
        #     OR (child:Placement AND parent:Interview)
        # CALL apoc.merge.relationship(
        #     parent,
        #     CASE 
        #         WHEN parent:ResearchContact THEN 'HANDLES'  # Use correct relationship type
        #         ELSE 'CHILD_OF' 
        #     END,
        #     {},
        #     {},
        #     child,
        #     {}
        # ) YIELD rel
        # SET rel.creation_date = datetime(),
        #     rel.last_updated = datetime(),
        #     rel.hierarchy_level = parent.pipeline_level
        # """
        
        # Update pipeline levels based on the depth of the hierarchy.
        level_query = """
        MATCH p=(start:ResearchContact)-[:CHILD_OF*]->(end)
        WITH nodes(p) as nodes
        UNWIND range(0, size(nodes)-1) as i
        WITH nodes[i] as node, i+1 as level
        SET node.pipeline_level = level
        """
        
        # Link supporting documents (Contract and Invoice) to Placement nodes.
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
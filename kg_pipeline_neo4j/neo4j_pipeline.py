import os
import json
import asyncio
from neo4j import GraphDatabase
from neo4j_graphrag.llm.openai import OpenAILLM
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

# 1. DATABASE & API CONFIGURATION
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_AUTH = ("neo4j", "your_password")
# Replace with your Azure or OpenAI credentials
AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# 2. DEFINING THE SCHEMA (Nodes with Descriptions and Properties)
# This replaces your 'entity_classes.json' and 'validator.py'
NODE_TYPES = [
    {
        "label": "Activity",
        "description": "Specific tasks, surveillance actions, or operational steps.",
        "properties": [
            {"name": "name", "type": "STRING", "required": True},
            {"name": "status", "type": "STRING", "description": "e.g., Active, Completed"},
            {"name": "frequency", "type": "STRING", "description": "e.g., Weekly, Monthly"}
        ]
    },
    {
        "label": "Process",
        "description": "High-level workflows or programs (e.g., Quality Management).",
        "properties": [
            {"name": "name", "type": "STRING", "required": True},
            {"name": "owner", "type": "STRING"}
        ]
    },
    {
        "label": "Document",
        "description": "Formal records, reports, or data outputs.",
        "properties": [
            {"name": "name", "type": "STRING", "required": True},
            {"name": "version", "type": "STRING"}
        ]
    },
    # Simple labels can still be added as strings
    "Actor", "Concept", "Deliverable", "Entity", "Event", "Procedure", "Role", "System"
]

RELATIONSHIP_TYPES = [
    "SUPPORTS", "OVERSEES", "GENERATES", "DEPENDS_ON", "PRODUCES"
]

# Patterns define allowed connections (Source -> Relation -> Target)
PATTERNS = [
    ("Activity", "SUPPORTS", "Process"),
    ("Role", "OVERSEES", "Activity"),
    ("System", "GENERATES", "Document"),
    ("Process", "PRODUCES", "Deliverable")
]

async def run_pipeline():
    # 3. INITIALIZE DRIVER & LLM
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    
    llm = OpenAILLM(
        model_name="gpt-4o",
        model_params={
            "api_key": AZURE_API_KEY,
            "azure_endpoint": AZURE_ENDPOINT,
            "api_version": "2024-08-01-preview"
        }
    )

    # 4. INITIALIZE THE KG BUILDER
    # 'additional_node_types=False' acts as your strict validator.
    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        schema={
            "node_types": NODE_TYPES,
            "relationship_types": RELATIONSHIP_TYPES,
            "patterns": PATTERNS,
            "additional_node_types": False,
            "additional_properties": True # Allows LLM to find extra context
        },
        perform_entity_resolution=True # Automatically merges similar entities
    )

    # 5. LOAD CHUNKS & EXECUTE
    try:
        with open("data/chunks.json") as f:
            chunks = json.load(f)
        
        print(f"🚀 Processing {len(chunks[:10])} chunks...")

        for chunk in chunks[:10]:
            print(f"📄 Ingesting Chunk: {chunk['chunk_id']}")
            
            # This extracts entities, properties, and relationships 
            # and performs a batch MERGE into Neo4j automatically.
            await kg_builder.run_async(text=chunk["text"])

        print("✅ Success! Your 1,400+ potential relations are being mapped.")

    except Exception as e:
        print(f"❌ Pipeline Error: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    asyncio.run(run_pipeline())
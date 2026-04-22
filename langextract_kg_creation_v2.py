import langextract as lx
import json

# --- STEP 1: Schema Mapping (The "Bridge") ---
# Defining the ontology instructions for the LLM
ontology_prompt = """
Extract entities and relationships based on the following ontology:
- Entity Types: Person, Organization
- Relationship: WORKS_AT (links Person to Organization)
"""

# --- STEP 2: Few-Shot Prompt Design (Teaching the "Skeleton") ---
examples = [
    lx.data.ExampleData(
        text="Alice Smith is a software engineer at Microsoft.",
        extractions=[
            lx.data.Extraction(extraction_class="Person", extraction_text="Alice Smith"),
            lx.data.Extraction(extraction_class="Organization", extraction_text="Microsoft"),
            lx.data.Extraction(
                extraction_class="WORKS_AT", 
                extraction_text="is a software engineer at",
                attributes={"subject": "Alice Smith", "object": "Microsoft"}
            )
        ]
    )
]

# --- STEP 3: Smart Chunking & Processing ---
raw_text = "Johan is working at Google."
extraction_result = lx.extract(
    text_or_documents=raw_text,
    prompt_description=ontology_prompt,
    examples=examples,
    model_id="gemini-2.0-flash"
)

# --- STEP 4: Grounding & Verification ---
# Ensuring each extraction has a valid source offset (provenance)
verified_extractions = []
for ext in extraction_result.extractions:
    # Check if the extraction has a source reference (grounding)
    if hasattr(ext, 'start_index'):
        print(f"Verified: {ext.extraction_class} found at {ext.start_index}")
        verified_extractions.append(ext)
    else:
        print(f"Warning: {ext.extraction_class} lacks grounding. Skipping.")

# --- STEP 5: Serialization for Graph Ingestion ---
# Converting verified extractions into Neo4j Cypher queries
cypher_commands = []
for ext in verified_extractions:
    if ext.extraction_class == "Person":
        cmd = f"MERGE (p:Person {{name: '{ext.extraction_text}'}})"
    elif ext.extraction_class == "Organization":
        cmd = f"MERGE (o:Organization {{name: '{ext.extraction_text}'}})"
    elif ext.extraction_class == "WORKS_AT":
        sub = ext.attributes.get('subject')
        obj = ext.attributes.get('object')
        cmd = f"MATCH (p:Person {{name: '{sub}'}}), (o:Organization {{name: '{obj}'}}) MERGE (p)-[:WORKS_AT]->(o)"
    
    cypher_commands.append(cmd)

# Final output
for command in cypher_commands:
    print(f"Executing Cypher: {command}")
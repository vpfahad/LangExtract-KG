import langextract as lx
from langextract.data import ExampleData, Extraction

# 1. Schema Mapping & Instruction
# We define our ontology rules clearly for the LLM
prompt = """
Extract entities and relationships based on the 'Organization Ontology'.
- Entities: Person, Organization
- Relations: WORKS_AT (links Person -> Organization)
"""

# 2. Few-Shot Prompt Design (Teaching the "Skeleton")
examples = [
    ExampleData(
        text="Fahad works at EY GDS.",
        extractions=[
            Extraction(extraction_class="Person", extraction_text="Fahad"),
            Extraction(extraction_class="Organization", extraction_text="EY GDS"),
            Extraction(extraction_class="WORKS_AT", extraction_text="works at", 
                       attributes={"subject": "Fahad", "object": "EY GDS"})
        ]
    )
]

# 3. Process with Multi-Pass (The Extraction Engine)
raw_text = "Sara is a developer at Google. She joined last month."
result = lx.extract(
    text_or_documents=raw_text,
    prompt_description=prompt,
    examples=examples,
    extraction_passes=2  # Ensures we don't miss links in multi-sentence text
)

# 4. Grounding & Verification
# Here we check if the extraction is linked to source text (char offsets)
print("--- Verification ---")
for ext in result.extractions:
    print(f"Extracted: {ext.extraction_text} (Class: {ext.extraction_class})")
    # This proves the data exists in the source text
    print(f"Source Trace: Found at character index {ext.start_offset}") 

# 5. Serialization for Graph Ingestion (The Loader)
print("\n--- Graph-Ready JSON for Neo4j ---")
graph_data = [ext.to_dict() for ext in result.extractions]
print(graph_data)
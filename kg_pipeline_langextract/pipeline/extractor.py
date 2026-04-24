#extractor.py
import os
import langextract as lx


# Ontology-aware prompt
PROMPT = """
Extract knowledge graph entities and relationships from the text.

Entity Classes:
Actor
Activity
Concept
Deliverable
Document
Entity
Event
Procedure
Process
Role
System

Relations:
SUPPORTS
OVERSEES
GENERATES
DEPENDS_ON
PRODUCES

Return entities and relations grounded in the text.
"""


# Few-shot examples help LangExtract learn structure
EXAMPLES = [
    lx.data.ExampleData(
        text="Engineering surveillance supports the Quality Management Programme.",
        extractions=[
            lx.data.Extraction(
                extraction_class="Activity",
                extraction_text="Engineering surveillance"
            ),
            lx.data.Extraction(
                extraction_class="Process",
                extraction_text="Quality Management Programme"
            ),
            lx.data.Extraction(
                extraction_class="Relation",
                extraction_text="supports",
                attributes={
                    "source": "Engineering surveillance",
                    "target": "Quality Management Programme",
                    "relation": "SUPPORTS"
                }
            )
        ]
    )
]


def extract_from_chunk(text):

    result = lx.extract(
        text_or_documents=text,
        prompt_description=PROMPT,
        examples=EXAMPLES,
        model_id="gpt-4o",
        fence_output=True,
        use_schema_constraints=False,
        api_key=os.getenv("AZURE_OPENAI_KEY")
    )

    entities = []
    relations = []

    for e in result.extractions:

        if e.char_interval is None:
            continue

        if e.extraction_class == "Relation":

            relations.append({
                "source": e.attributes.get("source"),
                "relation": e.attributes.get("relation"),
                "target": e.attributes.get("target")
            })

        else:

            entities.append({
                "name": e.extraction_text,
                "class": e.extraction_class
            })

    return {
        "entities": entities,
        "relations": relations
    }
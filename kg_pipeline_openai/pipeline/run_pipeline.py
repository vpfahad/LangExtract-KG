import json

from extractor import extract_entities_relations
from validator import validate_entities, validate_relations

# load ontology
with open("ontology/entity_classes.json") as f:
    entity_classes = json.load(f)["entity_classes"]

with open("ontology/relations.json") as f:
    relation_types = json.load(f)["relations"]

# load chunks
with open("data/chunks.json") as f:
    chunks = json.load(f)

chunks = chunks[:10]  # first 10 chunks

all_results = []

for chunk in chunks:

    print("Processing chunk:", chunk["chunk_id"])

    extraction = extract_entities_relations(
        chunk["text"],
        entity_classes,
        relation_types
    )

    entities = extraction["entities"]
    relations = extraction["relations"]

    valid_entities = validate_entities(entities, entity_classes)

    entity_names = [e["name"] for e in valid_entities]

    valid_relations = validate_relations(
        relations,
        relation_types,
        entity_names
    )

    result = {
        "chunk_id": chunk["chunk_id"],
        "entities": valid_entities,
        "relations": valid_relations
    }

    all_results.append(result)

# save validated triples

with open("validated_output.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("Pipeline completed.")
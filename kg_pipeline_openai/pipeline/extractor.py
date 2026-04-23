import langextract as lx
from config import client, MODEL

def create_extractor(entity_classes, relation_types):

    schema = {
        "entities": {
            "type": "list",
            "items": {
                "name": "string",
                "class": entity_classes
            }
        },
        "relations": {
            "type": "list",
            "items": {
                "source": "string",
                "relation": relation_types,
                "target": "string"
            }
        }
    }

    extractor = lx.Extractor(
        model=MODEL,
        client=client,
        schema=schema
    )

    return extractor


def extract_from_chunk(extractor, text):

    result = extractor.extract(text)

    return {
        "entities": result.get("entities", []),
        "relations": result.get("relations", [])
    }
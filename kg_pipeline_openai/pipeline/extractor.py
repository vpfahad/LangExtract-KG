import json
from config import client, MODEL_NAME


def extract_entities_relations(chunk, entity_classes, relation_types):

    prompt = f"""
You are an information extraction system.

Extract entities and relations from the text.

Entity Classes:
{entity_classes}

Relation Types:
{relation_types}

Return JSON format:

{{
 "entities":[
   {{"name":"","class":""}}
 ],
 "relations":[
   {{"source":"","relation":"","target":""}}
 ]
}}

Text:
{chunk}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    output = response.choices[0].message.content

    return json.loads(output)
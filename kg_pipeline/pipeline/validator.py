def validate_entities(entities, allowed_classes):

    valid_entities = []

    for e in entities:
        if e["class"] in allowed_classes:
            valid_entities.append(e)

    return valid_entities


def validate_relations(relations, allowed_relations, entity_names):

    valid_relations = []

    for r in relations:

        if r["relation"] not in allowed_relations:
            continue

        if r["source"] not in entity_names:
            continue

        if r["target"] not in entity_names:
            continue

        valid_relations.append(r)

    return valid_relations
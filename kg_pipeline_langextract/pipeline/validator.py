def validate_entities(entities, allowed_classes):

    valid = []

    for e in entities:
        if e["class"] in allowed_classes:
            valid.append(e)

    return valid


def validate_relations(relations, allowed_relations, entity_names):

    valid = []

    for r in relations:

        if r["relation"] not in allowed_relations:
            continue

        if r["source"] not in entity_names:
            continue

        if r["target"] not in entity_names:
            continue

        valid.append(r)

    return valid
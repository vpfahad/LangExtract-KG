import os
import textwrap
import langextract as lx
import langextract_azureopenai  # registers the AzureOpenAI provider plugin
import logging
import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph
from typing import List, Dict, Any, Optional
import json

# ─────────────────────────────────────────────
# 1. Azure OpenAI model config builder
# ─────────────────────────────────────────────

def build_azure_model_config() -> lx.factory.ModelConfig:
    """
    Build a LangExtract ModelConfig that points at the Azure OpenAI GPT-4o deployment.
    Reads credentials from environment variables set by load_azure_credentials().
    """
    return lx.factory.ModelConfig(
        model_id=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],   # e.g. "gpt-4o"
        provider="AzureOpenAILanguageModel",
        provider_kwargs={
            "api_key":        os.environ["AZURE_OPENAI_API_KEY"],
            "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
            "api_version":    os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        },
    )


# ─────────────────────────────────────────────
# 2. Entity / Relationship Extractor
# ─────────────────────────────────────────────

def document_extractor_tool(unstructured_text: str, user_query: str) -> dict:
    """
    Extracts structured information from unstructured text based on a user query.
    Uses LangExtract + Azure OpenAI GPT-4o with dynamic few-shot example selection.
    """
    prompt = textwrap.dedent(f"""
    You are an expert at extracting specific information from documents.
    Based on the user's query, extract the relevant information from the provided text.
    The user's query is: "{user_query}"
    Provide the output in a structured JSON format.
    """)

    # ── Dynamic few-shot example selection ──
    examples = []
    query_lower = user_query.lower()

    if any(kw in query_lower for kw in ["financial", "revenue", "company", "fiscal"]):
        examples.append(lx.data.ExampleData(
            text="In Q1 2023, Innovate Inc. reported a revenue of $15 million.",
            extractions=[
                lx.data.Extraction(extraction_class="company_name",
                                   extraction_text="Innovate Inc.",
                                   attributes={"name": "Innovate Inc."}),
                lx.data.Extraction(extraction_class="revenue",
                                   extraction_text="$15 million",
                                   attributes={"value": 15000000, "currency": "USD"}),
                lx.data.Extraction(extraction_class="fiscal_period",
                                   extraction_text="Q1 2023",
                                   attributes={"period": "Q1 2023"}),
            ]
        ))

    elif any(kw in query_lower for kw in ["legal", "agreement", "parties", "effective date"]):
        examples.append(lx.data.ExampleData(
            text="This agreement is between John Doe and Jane Smith, effective 2024-01-01.",
            extractions=[
                lx.data.Extraction(extraction_class="party",
                                   extraction_text="John Doe",
                                   attributes={"name": "John Doe"}),
                lx.data.Extraction(extraction_class="party",
                                   extraction_text="Jane Smith",
                                   attributes={"name": "Jane Smith"}),
                lx.data.Extraction(extraction_class="effective_date",
                                   extraction_text="2024-01-01",
                                   attributes={"date": "2024-01-01"}),
            ]
        ))

    elif any(kw in query_lower for kw in ["social", "post", "feedback", "restaurant"]):
        examples.append(lx.data.ExampleData(
            text="I tried the new 'Taste Lover' restaurant in TST today. The black truffle risotto was amazing, but the Tiramisu was just average.",
            extractions=[
                lx.data.Extraction(extraction_class="restaurant_name",
                                   extraction_text="Taste Lover",
                                   attributes={"name": "Taste Lover"}),
                lx.data.Extraction(extraction_class="dish",
                                   extraction_text="black truffle risotto",
                                   attributes={"name": "black truffle risotto", "sentiment": "positive"}),
                lx.data.Extraction(extraction_class="dish",
                                   extraction_text="Tiramisu",
                                   attributes={"name": "Tiramisu", "sentiment": "neutral"}),
            ]
        ))

    else:  # generic fallback
        examples.append(lx.data.ExampleData(
            text="Juliet looked at Romeo with a sense of longing.",
            extractions=[
                lx.data.Extraction(extraction_class="character",
                                   extraction_text="Juliet",
                                   attributes={"name": "Juliet"}),
                lx.data.Extraction(extraction_class="character",
                                   extraction_text="Romeo",
                                   attributes={"name": "Romeo"}),
                lx.data.Extraction(extraction_class="emotion",
                                   extraction_text="longing",
                                   attributes={"type": "longing"}),
            ]
        ))

    logging.info(f"Selected {len(examples)} few-shot example(s).")

    model_config = build_azure_model_config()

    result = lx.extract(
        text_or_documents=unstructured_text,
        prompt_description=prompt,
        examples=examples,
        model_config=model_config,   # ← Azure GPT-4o instead of Gemini
        fence_output=True,           # required for OpenAI-family models
        use_schema_constraints=False,
    )

    logging.info(f"Extraction result: {result}")

    extractions = [
        {"text": e.extraction_text, "class": e.extraction_class, "attributes": e.attributes}
        for e in result.extractions
    ]

    return {"extracted_data": extractions}


# ─────────────────────────────────────────────
# 3. Azure Credentials Loader
# ─────────────────────────────────────────────

def load_azure_credentials() -> bool:
    """
    Load Azure OpenAI credentials from secrets.toml or sidebar inputs.
    Returns True when all required fields are present.
    """
    st.sidebar.header("🔑 Azure OpenAI Credentials")

    secrets_file = os.path.join(".streamlit", "secrets.toml")
    from_secrets = (
        os.path.exists(secrets_file)
        and "AZURE_OPENAI_API_KEY" in st.secrets
    )

    if from_secrets:
        os.environ["AZURE_OPENAI_API_KEY"]        = st.secrets["AZURE_OPENAI_API_KEY"]
        os.environ["AZURE_OPENAI_ENDPOINT"]       = st.secrets["AZURE_OPENAI_ENDPOINT"]
        os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]= st.secrets.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        os.environ["AZURE_OPENAI_API_VERSION"]    = st.secrets.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        st.sidebar.success("✅ Credentials loaded from secrets.toml")
        return True

    # Manual input fallback
    api_key    = st.sidebar.text_input("API Key",      type="password", placeholder="••••••••")
    endpoint   = st.sidebar.text_input("Endpoint URL", placeholder="https://YOUR-RESOURCE.openai.azure.com/")
    deployment = st.sidebar.text_input("Deployment Name", value="gpt-4o",         placeholder="gpt-4o")
    api_ver    = st.sidebar.text_input("API Version",  value="2024-12-01-preview", placeholder="2024-12-01-preview")

    if api_key and endpoint and deployment:
        os.environ["AZURE_OPENAI_API_KEY"]        = api_key
        os.environ["AZURE_OPENAI_ENDPOINT"]       = endpoint
        os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]= deployment
        os.environ["AZURE_OPENAI_API_VERSION"]    = api_ver or "2024-12-01-preview"
        st.sidebar.success("✅ Credentials accepted")
        return True

    st.sidebar.warning("⚠️ Fill in all Azure credentials to continue.")


# ─────────────────────────────────────────────
# 3. Graph Visualisation Helpers
# ─────────────────────────────────────────────

def format_output_agraph(output: dict):
    """Convert raw graph dict → agraph Node / Edge objects."""
    nodes = []
    edges = []

    for node in output["nodes"]:
        nodes.append(Node(id=node["id"], label=node["label"], size=8, shape="diamond"))

    for edge in output["edges"]:
        edges.append(Edge(
            source=edge["source"],
            label=edge["relation"],
            target=edge["target"],
            color="#4CAF50",
            arrows="to"
        ))

    return nodes, edges


def display_agraph(nodes, edges):
    """Render the interactive agraph component."""
    config = Config(
        width=950,
        height=950,
        directed=True,
        physics=True,
        hierarchical=True,
        nodeHighlightBehavior=False,
        highlightColor="#F7A7A6",
        collapsible=False,
        node={"labelProperty": "label"},
    )
    return agraph(nodes=nodes, edges=edges, config=config)


# ─────────────────────────────────────────────
# 4. Core GraphRAG Pipeline Functions
# ─────────────────────────────────────────────

def extract_entities(documents: List[str]) -> List[Dict[str, Any]]:
    """Extract named entities from all documents."""
    all_entities = []
    for doc in documents:
        result = document_extractor_tool(
            doc,
            "Extract financial entities including company names, revenue figures, "
            "and fiscal periods from business documents"
        )
        all_entities.extend(result["extracted_data"])
    return all_entities


def extract_relationships(documents: List[str]) -> List[Dict[str, Any]]:
    """Extract relationships between entities from all documents."""
    all_relationships = []
    for doc in documents:
        result = document_extractor_tool(
            doc,
            "Extract financial relationships and revenue connections "
            "between companies and fiscal periods"
        )
        all_relationships.extend(result["extracted_data"])
    return all_relationships


def build_graph_data(
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Build node/edge lists for the graph visualisation."""
    nodes = []
    edges = []

    # Create nodes
    entity_map: Dict[str, str] = {}
    for i, entity in enumerate(entities):
        node_id = str(i)
        nodes.append({"id": node_id, "label": entity["text"], "type": entity["class"]})
        entity_map[entity["text"].lower()] = node_id

    # Create edges based on co-occurrence within relationship spans
    for rel in relationships:
        rel_text = rel["text"].lower()
        found = [eid for etext, eid in entity_map.items() if etext in rel_text]

        for i in range(len(found)):
            for j in range(i + 1, len(found)):
                edges.append({
                    "source": found[i],
                    "target": found[j],
                    "relation": rel["class"]
                })

    # Fallback: connect every entity pair if no explicit edges found
    if not edges:
        st.write("ℹ️ No relationship edges found — creating fallback co-occurrence edges.")
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                edges.append({"source": str(i), "target": str(j), "relation": "related_to"})

    return {"nodes": nodes, "edges": edges}


def answer_query(
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    query: str
) -> Optional[Dict[str, Any]]:
    """Filter entities and relationships that match the query keywords."""
    if not query:
        return None

    words = query.split()

    relevant_entities = [
        e for e in entities
        if any(w.lower() in e["text"].lower() or w.lower() in str(e["attributes"]).lower()
               for w in words)
    ]
    relevant_relationships = [
        r for r in relationships
        if any(w.lower() in r["text"].lower() or w.lower() in str(r["attributes"]).lower()
               for w in words)
    ]

    return {
        "query": query,
        "relevant_entities": relevant_entities,
        "relevant_relationships": relevant_relationships,
        "entity_count": len(relevant_entities),
        "relationship_count": len(relevant_relationships),
    }


def process_documents(
    documents: List[str],
    query: str = None
) -> Dict[str, Any]:
    """
    Full pipeline:
      1. Extract entities
      2. Extract relationships
      3. Build graph data
      4. (optionally) answer a query
    """
    entities = extract_entities(documents)
    relationships = extract_relationships(documents)

    st.write(f"🔍 Found **{len(entities)}** entities and **{len(relationships)}** relationships.")

    graph_data = build_graph_data(entities, relationships)

    st.write(
        f"🕸️ Graph has **{len(graph_data['nodes'])}** nodes "
        f"and **{len(graph_data['edges'])}** edges."
    )

    results = answer_query(entities, relationships, query) if query else None

    return {
        "entities": entities,
        "relationships": relationships,
        "graph_data": graph_data,
        "results": results,
    }


# ─────────────────────────────────────────────
# 5. Streamlit UI
# ─────────────────────────────────────────────

def main():
    st.set_page_config(page_title="GraphRAG with LangExtract × Azure GPT-4o", layout="wide")
    st.title("🧠 GraphRAG with LangExtract × Azure OpenAI GPT-4o")
    st.caption("Google's open-source LangExtract + Azure OpenAI GPT-4o + Streamlit knowledge-graph pipeline")

    # ── Sidebar: Azure credentials ──
    credentials_ok = load_azure_credentials()

    if not credentials_ok:
        st.warning("⚠️ Please fill in your Azure OpenAI credentials in the sidebar to continue.")
        st.stop()

    # ── Predefined tech-company documents ──
    default_documents = [
        "Apple Inc. was founded by Steve Jobs and Steve Wozniak in 1976. "
        "The company is headquartered in Cupertino, California. "
        "Steve Jobs served as CEO until his death in 2011.",

        "Microsoft Corporation was founded by Bill Gates and Paul Allen in 1975. "
        "It's based in Redmond, Washington. Bill Gates was the CEO for many years.",

        "Both Apple and Microsoft are major technology companies that compete in various "
        "markets including operating systems and productivity software. "
        "They have a long history of rivalry.",

        "Google was founded by Larry Page and Sergey Brin in 1998. "
        "The company started as a search engine but has expanded into many areas "
        "including cloud computing and artificial intelligence.",
    ]

    # ── Custom document input ──
    st.subheader("📄 Documents")
    custom_doc = st.text_area(
        "Add a custom document (optional — leave blank to use the defaults):",
        height=100,
        placeholder="e.g. Amazon was founded by Jeff Bezos in 1994 in Seattle, Washington..."
    )

    documents = default_documents.copy()
    if custom_doc.strip():
        documents.append(custom_doc.strip())

    st.success(f"Using **{len(documents)}** document(s).")

    with st.expander("📋 View documents"):
        for i, doc in enumerate(documents, 1):
            st.markdown(f"**Doc {i}:** {doc}")

    # ── Optional query ──
    query = st.text_input(
        "🔎 Enter a query to filter results (optional):",
        placeholder="e.g. Who founded Apple?"
    )

    # ── Run pipeline ──
    if st.button("🚀 Process Documents & Build Graph", type="primary"):
        with st.spinner("Extracting entities and relationships…"):
            result = process_documents(documents, query if query else None)

        tab1, tab2, tab3, tab4 = st.tabs([
            "🕸️ Graph Visualisation",
            "🏷️ Entities",
            "🔗 Relationships",
            "❓ Query Results",
        ])

        with tab1:
            st.subheader("Interactive Knowledge Graph")
            nodes, edges = format_output_agraph(result["graph_data"])
            if nodes:
                display_agraph(nodes, edges)
            else:
                st.info("No graph data to display.")

        with tab2:
            st.subheader("Extracted Entities")
            if result["entities"]:
                for entity in result["entities"]:
                    with st.expander(f"{entity['text']}  ·  `{entity['class']}`"):
                        st.json(entity["attributes"])
            else:
                st.info("No entities extracted.")

        with tab3:
            st.subheader("Extracted Relationships")
            if result["relationships"]:
                for rel in result["relationships"]:
                    with st.expander(f"{rel['text']}  ·  `{rel['class']}`"):
                        st.json(rel["attributes"])
            else:
                st.info("No relationships extracted.")

        with tab4:
            if query and result["results"]:
                st.subheader(f"Results for: *{query}*")
                st.json(result["results"])
            else:
                st.info("No query provided or no matching results.")


if __name__ == "__main__":
    main()

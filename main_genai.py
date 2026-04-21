"""
Simple Knowledge Graph Pipeline using Google LangExtract + Azure OpenAI

Install:
    pip install langextract langextract-azureopenai networkx matplotlib

Set these env vars (or fill in directly below):
    AZURE_OPENAI_API_KEY      = your Azure OpenAI key
    AZURE_OPENAI_ENDPOINT     = https://your-resource.openai.azure.com/
    AZURE_OPENAI_API_VERSION  = 2024-12-01-preview
    AZURE_OPENAI_DEPLOYMENT   = your deployment name (e.g. gpt-4o)
"""

import os
import langextract as lx
import networkx as nx
import matplotlib.pyplot as plt

# ── 1. Configure Azure OpenAI credentials ────────────────────────────────────
os.environ["AZURE_OPENAI_API_KEY"]     = "your_azure_openai_key_here"
os.environ["AZURE_OPENAI_ENDPOINT"]    = "https://your-resource.openai.azure.com/"
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-12-01-preview"
DEPLOYMENT_NAME = "gpt-4o"   # your Azure deployment name

# ── 2. Build ModelConfig for Azure OpenAI ────────────────────────────────────
azure_config = lx.factory.ModelConfig(
    model_id=DEPLOYMENT_NAME,
    provider="AzureOpenAILanguageModel",   # registered by langextract-azureopenai
    provider_kwargs={
        "api_key":        os.getenv("AZURE_OPENAI_API_KEY"),
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version":    os.getenv("AZURE_OPENAI_API_VERSION"),
    },
)

# ── 3. Hardcoded entities & relations (your final knowledge graph data) ───────
ENTITIES = [
    {"name": "Google",       "type": "Company"},
    {"name": "Sundar Pichai","type": "Person"},
    {"name": "DeepMind",     "type": "Company"},
    {"name": "Gemini",       "type": "Product"},
    {"name": "Python",       "type": "Language"},
    {"name": "USA",          "type": "Location"},
]

RELATIONS = [
    ("Sundar Pichai", "CEO_of",     "Google"),
    ("Google",        "owns",       "DeepMind"),
    ("Google",        "created",    "Gemini"),
    ("Google",        "located_in", "USA"),
    ("Gemini",        "built_with", "Python"),
    ("DeepMind",      "developed",  "Gemini"),
]

# ── 4. Use LangExtract to process entities via Azure OpenAI ───────────────────
entity_text = "\n".join(
    f"{e['name']} is a {e['type']}." for e in ENTITIES
) + "\n" + "\n".join(
    f"{s} {r.replace('_', ' ')} {o}." for s, r, o in RELATIONS
)

prompt = "Extract all entities with their type (Company, Person, Product, Language, Location)."

examples = [
    lx.data.ExampleData(
        text="OpenAI is a Company. Sam Altman is a Person.",
        extractions=[
            lx.data.Extraction(extraction_class="Company", extraction_text="OpenAI"),
            lx.data.Extraction(extraction_class="Person",  extraction_text="Sam Altman"),
        ],
    )
]

print("Running LangExtract with Azure OpenAI...")
result = lx.extract(
    text_or_documents=entity_text,
    prompt_description=prompt,
    examples=examples,
    model_config=azure_config,    # ← pass ModelConfig instead of model_id
    fence_output=True,            # required for OpenAI-based models
    use_schema_constraints=False,
)
print(f"LangExtract found {len(result.extractions)} extractions.\n")

# ── 5. Build Knowledge Graph with NetworkX ────────────────────────────────────
G = nx.DiGraph()

for entity in ENTITIES:
    G.add_node(entity["name"], label=entity["type"])

for subject, relation, obj in RELATIONS:
    G.add_edge(subject, obj, label=relation)

# ── 6. Visualize ──────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 7))
pos = nx.spring_layout(G, seed=42)

color_map = {
    "Company": "#4A90D9", "Person": "#E67E22",
    "Product": "#2ECC71", "Language": "#9B59B6", "Location": "#E74C3C",
}
node_colors = [color_map.get(G.nodes[n].get("label", ""), "#AAA") for n in G.nodes]

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000)
nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")
nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20, edge_color="#555",
                       connectionstyle="arc3,rad=0.1")
nx.draw_networkx_edge_labels(G, pos,
    edge_labels={(u, v): d["label"] for u, v, d in G.edges(data=True)},
    font_size=7, font_color="darkred")

plt.title("Knowledge Graph (LangExtract + Azure OpenAI + NetworkX)", fontsize=13)
plt.axis("off")
plt.tight_layout()
plt.savefig("knowledge_graph.png", dpi=150)
plt.show()
print("Graph saved as knowledge_graph.png")

# ── 7. Save LangExtract output to JSONL + HTML visualization ─────────────────
lx.io.save_annotated_documents([result], output_name="kg_extractions.jsonl")
html = lx.visualize("kg_extractions.jsonl")
with open("kg_visualization.html", "w") as f:
    f.write(html)
print("LangExtract visualization saved as kg_visualization.html")
# GraphRAG with LangExtract × Azure OpenAI GPT-4o

Recreates the pipeline from the YouTube video  
**"LangExtract + Knowledge Graph — Google's New Library for NLP Tasks"**,  
using **Azure OpenAI GPT-4o** instead of Google Gemini.

---

## What it does

1. **Extracts entities** from text using Google's **LangExtract** + Azure GPT-4o.
2. **Extracts relationships** between entities using dynamic few-shot prompting.
3. **Builds a knowledge graph** (nodes = entities, edges = relationships).
4. **Visualises** the graph interactively inside Streamlit via `streamlit-agraph`.
5. **Answers optional queries** by filtering the graph for relevant results.

---

## Folder structure

```
langextract-kg/
├── app.py
├── requirements.txt
└── .streamlit/
    └── secrets.toml   ← put Azure credentials here
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your Azure OpenAI credentials

You need 4 values from [Azure AI Studio](https://oai.azure.com/):

| Value | Where to find it |
|---|---|
| API Key | Azure Portal → OpenAI resource → Keys and Endpoint |
| Endpoint URL | Same page — `https://YOUR-NAME.openai.azure.com/` |
| Deployment Name | Azure AI Studio → Deployments (name of your GPT-4o deployment) |
| API Version | Use `2024-12-01-preview` |

**Option A — secrets.toml:**
```toml
AZURE_OPENAI_API_KEY         = "abc123..."
AZURE_OPENAI_ENDPOINT        = "https://my-resource.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o"
AZURE_OPENAI_API_VERSION     = "2024-12-01-preview"
```

**Option B — sidebar at runtime:** leave the file as-is and fill in the boxes.

### 3. Run
```bash
streamlit run app.py
```

---

## Key change from the Gemini version

| | Gemini | Azure GPT-4o |
|---|---|---|
| Package | `langextract` only | + `langextract-azureopenai` plugin |
| Auth | `api_key=GOOGLE_API_KEY` | `model_config=lx.factory.ModelConfig(...)` |
| Extra flags | — | `fence_output=True`, `use_schema_constraints=False` |

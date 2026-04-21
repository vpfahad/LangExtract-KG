# Knowledge Graph Pipeline — Setup Guide

A simple pipeline using **Google LangExtract** + **Azure OpenAI** to build a knowledge graph, visualized with NetworkX and Matplotlib.

---

## Prerequisites

- Python **3.10+**
- A virtual environment (recommended)
- An **Azure OpenAI** resource with a deployed model (e.g. `gpt-4o`)

---

## 1. Clone / Create the Project Folder

```bash
mkdir KG-LangExtract
cd KG-LangExtract
```

Place `main.py` (the pipeline script) inside this folder.

---

## 2. Create a Virtual Environment

**Using `venv`:**
```bash
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows
```

**Using `uv` (faster alternative):**
```bash
uv venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows
```

---

## 3. Install Dependencies

```bash
pip install langextract langextract-azureopenai networkx matplotlib python-dotenv
```

Or with `uv`:
```bash
uv pip install langextract langextract-azureopenai networkx matplotlib python-dotenv
```

| Package | Purpose |
|---|---|
| `langextract` | Google's structured extraction library |
| `langextract-azureopenai` | Azure OpenAI provider plugin for LangExtract |
| `networkx` | Build and manage the knowledge graph |
| `matplotlib` | Visualize the graph |
| `python-dotenv` | Load credentials from `.env` file |

---

## 4. Create the `.env` File

In the **same folder** as `main.py`, create a file named `.env`:

```env
AZURE_OPENAI_API_KEY=your_azure_openai_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

> **Where to find these values:**
> - Go to [Azure Portal](https://portal.azure.com) → your Azure OpenAI resource
> - **API Key:** Keys and Endpoint → Key 1
> - **Endpoint:** Keys and Endpoint → Endpoint
> - **Deployment name:** Azure OpenAI Studio → Deployments

> ⚠️ Never commit your `.env` file to Git. Add it to `.gitignore`:
> ```bash
> echo ".env" >> .gitignore
> ```

---

## 5. Project Structure

After setup, your folder should look like this:

```
KG-LangExtract/
├── main.py
├── .env
├── .gitignore
└── .venv/
```

---

## 6. Run the Pipeline

```bash
python main.py
```

Or with `uv`:
```bash
uv run main.py
```

---

## 7. Output Files

After a successful run, three output files are generated in the project folder:

| File | Description |
|---|---|
| `knowledge_graph.png` | Static image of the knowledge graph |
| `kg_extractions.jsonl` | LangExtract structured extraction results |
| `kg_visualization.html` | Interactive HTML visualization — open in any browser |

---

## Troubleshooting

**`AttributeError: module 'langextract' has no attribute 'ExampleData'`**
Use `lx.data.ExampleData` and `lx.data.Extraction` — not `lx.ExampleData`.

**`langextract has no attribute 'factory'` or provider not found**
Make sure `langextract-azureopenai` is installed. It registers the `AzureOpenAILanguageModel` provider at import time.

**`AuthenticationError` or `401 Unauthorized`**
Double-check your `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` in the `.env` file.

**`DeploymentNotFound` or `404`**
Make sure `AZURE_OPENAI_DEPLOYMENT` exactly matches the deployment name in Azure OpenAI Studio (it's case-sensitive).

**`.env` values not loading**
Ensure the `.env` file is in the same directory you're running the script from, and that `python-dotenv` is installed.

# StudyMate — Apodex MVP

A small educational RAG application using **LangGraph, two agents, ChromaDB, Sentence Transformers, Streamlit, and Apodex 1.1 Mini**.

## Workflow

PDF → chunks → embeddings → ChromaDB → LangGraph router → Tutor/Quiz Agent → Apodex → response

## Agents

- **Tutor Agent**: explains concepts using retrieved study material.
- **Quiz Agent**: creates MCQs using retrieved study material.

## Run

```bash
python -m venv .venv
# activate the environment
pip install -r requirements.txt
cp .env.example .env
# add your APODEX_API_KEY
streamlit run app.py
```

Apodex core models expose an OpenAI-compatible Chat Completions API at `https://api.apodex.ai/v1`, and the model ID is `apodex-1.1-mini`.

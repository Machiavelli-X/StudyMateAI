# StudyMate AI 📚

An AI-powered study assistant. Upload your lecture notes (PDF), then summarize them, ask questions, take quizzes, and get personalized learning reflections — all grounded in *your* material. The Ask AI feature runs a **self-verifying agent** that cross-checks its answer against the live web and corrects itself.

Built with **Streamlit**, **ChromaDB**, **sentence-transformers**, **LangChain**, **OpenRouter**, and **Tavily**.

---

## Features

| Page | What it does |
|------|--------------|
| **Upload** | Turn a PDF into searchable study material |
| **Summary** | Generate concise, exam-focused notes |
| **Ask AI** | Answer questions from your notes, then web-verify & self-correct (agentic) |
| **Quiz** | Generate MCQs, take them, get a full review with explanations |
| **Reflection** | Get a personalized learning report after a quiz |

---

## How it works (the workflow)

### 1. Upload — from PDF to searchable notes

1. **Extract text** from the PDF — [PyMuPDF](https://pymupdf.readthedocs.io/)
2. **Split into chunks** — LangChain `RecursiveCharacterTextSplitter` (500 chars, 100 overlap)
3. **Convert chunks to vectors (embeddings)** — `sentence-transformers/all-MiniLM-L6-v2`
4. **Store the vectors** — [ChromaDB](https://www.trychroma.com/) (persistent, on disk)

> Embeddings are lists of numbers that capture meaning. They let the app find the notes most relevant to any question.

### 2. Summary

Reads the stored chunks and asks an LLM to produce a structured summary: **Summary → Key Points → Important Concepts → Exam Tips**.

### 3. Ask AI — the agentic loop ⭐

A **self-reflective RAG agent**. It doesn't just answer — it verifies and corrects itself:

| Step | Action | Technology |
|------|--------|------------|
| 1. Retrieve | Find the most relevant chunks of your notes | ChromaDB + embeddings |
| 2. Draft | Write an answer using **only** your notes | LLM (LangChain) |
| 3. Web search | Search the live web for the topic | [Tavily](https://tavily.com) |
| 4. Validate | Compare the draft against the web, point-by-point | LLM (LangChain) |
| 5. Revise | Regenerate the answer with corrections & additions | LLM (LangChain) |

Each check is labelled ✅ *validated*, ⚠️ *corrected*, or ➕ *added* (missing from the notes) — so you can watch the agent catch and fix its own mistakes.

### 4. Quiz

Generates multiple-choice questions **only** from your notes, grades your answers, and produces a full review: an explanation for every question plus an overall learning report (strengths, weak areas, topics to revise).

### 5. Reflection

The Quiz step sends a structured result (score, weak/strong topics) to a **Reflection agent**, which writes a personalized study plan: overall performance, strengths, weak areas, topics to revise, recommendations, and motivation.

---

## Tech stack

| Phase | Technology |
|-------|------------|
| UI | Streamlit |
| PDF text extraction | PyMuPDF |
| Text chunking | LangChain text splitters |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector storage & search | ChromaDB |
| LLM access | LangChain (`langchain-openai`) |
| LLM provider | OpenRouter (Nemotron) — Apodex / Groq optional |
| Web search (agent validation) | Tavily |
| Config / secrets | python-dotenv (`.env`) |

---

## Run it locally

```bash
# 1. Create & activate a virtual environment
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your keys
cp .env.example .env
#    then open .env and add:
#    OPENROUTER_API_KEY=...   (your LLM key)
#    TAVILY_API_KEY=...       (your web-search key, from tavily.com)

# 4. Run
streamlit run app.py
```

The app opens in your browser. Upload a PDF, then try Summary, Ask AI, Quiz, and Reflection.

> **Note:** the first page load can take ~15–25 seconds while the embedding model loads into memory (once). After that, pages are fast.

---

## Project structure

```
├── app.py                  # Entry point + navigation
├── config.py               # Settings (env-driven)
├── pages/                  # Streamlit pages: Upload, Summary, Ask AI, Quiz, Reflection
├── services/               # Business logic (one service per feature)
│   └── research_service.py # The self-reflective agentic loop (Ask AI)
├── rag/                    # Retrieval: chunker, embeddings, vector store
├── agents/                 # LLM provider routing + inter-agent messages
├── components/             # CSS / styling
└── utils/                  # PDF loader
```

# 🎓 StudyMate AI — Interview Preparation Guide

> Your complete, honest, project-specific guide to explaining StudyMate AI in an interview.  
> Covers: What it is → Why it works → How every part is built → Expected questions + model answers.

---

## PART 1 — What is StudyMate AI?

**One-liner (memorize this):**
> "StudyMate AI is a multi-agent AI study assistant that lets students upload lecture PDFs and then get AI-generated summaries, take quizzes, ask questions, and receive learning reflections — all powered by a RAG pipeline with two specialized LLMs."

**What problems does it solve?**
- Students waste time re-reading entire notes to find key concepts
- Traditional study tools don't personalize to *your* uploaded document
- Quiz generation is manual and time-consuming

**The core idea:**  
Upload a PDF → AI reads, understands, and stores it → You interact with it through 4 AI-powered features

---

## PART 2 — Technology Stack (Know Each One)

| Technology | What it does in YOUR project |
|---|---|
| **Python** | Core language for all backend logic |
| **Streamlit** | The web UI framework — builds all 5 pages |
| **LangChain** | Connects LLMs to your app — handles prompt building and LLM calls |
| **LangGraph** | Builds the multi-node AI workflow (route → retrieve → agent) |
| **ChromaDB** | Vector database — stores document embeddings on disk |
| **SentenceTransformers** | Converts text to embeddings using `all-MiniLM-L6-v2` |
| **PyMuPDF (fitz)** | Extracts text from uploaded PDF files |
| **OpenRouter API** | Gateway to run free LLMs like Nemotron without paying per-call |
| **dotenv** | Loads API keys from `.env` file securely |

---

## PART 3 — Architecture Deep Dive

### The Big Picture (say this first in any architecture question)

```
[User uploads PDF]
        ↓
[DocumentService: Extract text → Chunk → Embed → Store in ChromaDB]
        ↓
[User picks a feature: Summary / Quiz / Ask AI / Reflection]
        ↓
[Service Layer fetches relevant chunks from ChromaDB]
        ↓
[AIProvider sends chunks + prompt to LLM via OpenRouter]
        ↓
[LLM returns answer → UI displays it]
```

### Folder Structure and What Each Does

```
StudyMate-Apodex-MVP/
│
├── app.py               ← Entry point. Sets up Streamlit navigation
├── config.py            ← All settings: API keys, paths, chunk sizes
├── .env                 ← Secret keys (never committed to git)
│
├── pages/               ← One file per UI page (Streamlit multi-page)
│   ├── Upload.py        ← File upload + processing trigger
│   ├── Summary.py       ← Calls SummaryService, shows + downloads summary
│   ├── Quiz.py          ← Quiz UI, answer submission, score review
│   ├── Ask_AI.py        ← Chat interface using RAG
│   └── Reflection.py    ← Learning reflection and feedback
│
├── services/            ← Business logic layer (one service per feature)
│   ├── document_service.py   ← Full pipeline: save → extract → chunk → embed → store
│   ├── summary_service.py    ← Fetches all chunks, sends to LLM for summary
│   ├── quiz_service.py       ← Generates MCQ quiz, then reviews answers
│   ├── chat_service.py       ← RAG-based Q&A
│   └── reflection_service.py ← Generates learning reflection
│
├── agents/
│   ├── ai_provider.py   ← Multi-model router (Model 1 vs Model 2)
│   └── messages.py      ← Prompt/message builders
│
├── rag/                 ← Everything related to Retrieval-Augmented Generation
│   ├── chunker.py       ← Splits text into 500-char chunks (100 overlap)
│   ├── embeddings.py    ← Converts chunks to vectors using MiniLM
│   ├── vector_store.py  ← ChromaDB wrapper: store + search embeddings
│   └── document_service.py ← Low-level PDF text extraction
│
└── graph/               ← LangGraph multi-agent workflow
    ├── state.py         ← Defines the shared state schema
    └── workflow.py      ← Builds the StateGraph with 4 nodes
```

---

## PART 4 — The RAG Pipeline (Most Important Concept)

**What is RAG?**  
RAG = Retrieval-Augmented Generation.  
Instead of asking an LLM from memory, you *first retrieve* relevant text from your document, then *give that text to the LLM* as context. This makes answers grounded in the actual document.

**Why not just send the whole PDF to the LLM?**  
- LLMs have token limits (can't fit 50-page PDFs)
- Sending everything is slow and expensive
- RAG finds only the *relevant* chunks, making it faster and more accurate

### Step-by-step RAG pipeline in your code:

**STEP 1 — PDF Text Extraction**
```python
# services/document_service.py
import fitz  # PyMuPDF
text = self.loader.extract_text(str(file_path))
```
PyMuPDF opens the PDF and reads raw text from each page.

**STEP 2 — Chunking**
```python
# rag/chunker.py
RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
```
The text is split into chunks of **500 characters** with **100 character overlap**.  
The overlap ensures that if a concept spans a chunk boundary, it won't be lost.

**STEP 3 — Embedding**
```python
# rag/embeddings.py
SentenceTransformer("all-MiniLM-L6-v2")
```
Each chunk is converted into a **384-dimensional vector** (a list of 384 numbers) that captures its meaning. Similar text → similar vectors.

**STEP 4 — Storage**
```python
# rag/vector_store.py
self.collection.add(ids=ids, embeddings=embeddings, documents=chunks)
```
All vectors + original text are saved to **ChromaDB** on disk at `data/chroma_db/`.

**STEP 5 — Retrieval (at query time)**
```python
query_embedding = self.embedding_model.encode([query])[0].tolist()
results = self.collection.query(query_embeddings=[query_embedding], n_results=8)
```
The user's question is also embedded. ChromaDB finds the **8 most similar chunks** using cosine similarity.

**STEP 6 — Generation**  
The retrieved chunks are injected into a prompt and sent to the LLM:
```
"Here is the relevant study material: [chunks]
Answer the student's question: [question]"
```

---

## PART 5 — The LangGraph Multi-Agent Workflow

**What is LangGraph?**  
LangGraph is a library for building *stateful, graph-based AI workflows*. Instead of one LLM call, you define nodes (processing steps) and edges (connections between them). It's like a flowchart where each box is an AI agent.

### Your workflow in `graph/workflow.py`:

```
START → [route] → [retrieve] → [explain]   → END
                             ↘ [flashcard] → END
```

**Node 1: `route`**  
Looks at the user's question and decides intent.  
If the question contains words like "flashcard", "memorize", "anki" → `flashcard` intent  
Otherwise → `explain` intent

**Node 2: `retrieve`**  
Calls ChromaDB to get the top-8 most relevant chunks for the question.  
Stores them in the shared `StudyState`.

**Node 3: `explain_agent`**  
Uses **Model 1** to explain the concept from retrieved context.  
Prompt: "Explain like I'm 5, using a fun analogy"

**Node 4: `flashcard_agent`**  
Uses **Model 2** to generate 5 Anki-style flashcards as JSON.  
Returns: `[{"front": "...", "back": "..."}, ...]`

**The shared state (`graph/state.py`):**  
All nodes read from and write to a shared Python TypedDict:
```
question, intent, context, sources, answer
```
This is how nodes communicate without calling each other directly.

---

## PART 6 — Multi-Model Routing (`agents/ai_provider.py`)

**Why two models?**  
Different tasks need different capabilities:
- **Summary and Quiz**: Need structured output, instruction-following → Model 1
- **Ask AI and Reflection**: Need deeper reasoning → Model 2

**How routing works:**
```python
TASK_PROVIDER_MAP = {
    "summary":  "model_1",
    "quiz":     "model_1",
    "question_answering": "model_2",
    "reflection":         "model_2",
}
```

When any service calls `ai_provider.get_llm(task="quiz")`, it automatically gets the right model. The UI doesn't know or care which model runs — the router handles it.

---

## PART 7 — Feature Walkthroughs

### Upload
1. User picks a PDF via Streamlit file uploader
2. `DocumentService.process()` is called
3. Pipeline runs: save → extract text → chunk → embed → store in ChromaDB
4. Result stored in `st.session_state` so other pages know a document is loaded

### Summary
1. `SummaryService` fetches **all** chunks from ChromaDB
2. Joins them into one long text (capped at 15,000 chars)
3. Sends to LLM with structured prompt asking for: Summary / Key Points / Important Concepts / Exam Tips
4. Result shown with `st.markdown()` and downloadable as `.txt`

### Quiz
1. Fetches all chunks, sends to LLM with strict JSON format requirements
2. LLM returns 5 MCQs: `[{question, options:{A,B,C,D}, answer, explanation}]`
3. User answers each question, submits
4. `review_quiz()` sends answers + correct answers back to LLM for a detailed review with score and per-question feedback

### Ask AI
1. User types a question
2. ChromaDB retrieves top-8 relevant chunks (RAG in action)
3. Chunks + question sent to LLM
4. Answer displayed with source chunks shown for transparency

### Reflection
1. Based on what the student has studied, LLM generates a learning reflection
2. Includes: what they learned, what needs more attention, suggested next steps

---

## PART 8 — Key Design Decisions

| Decision | Reason |
|---|---|
| **ChromaDB over Pinecone** | Local, no cloud cost, works offline, persistent on disk |
| **all-MiniLM-L6-v2** | Fast, small (80MB), runs locally without GPU, good accuracy |
| **chunk_size=500, overlap=100** | 500 chars ≈ 1-2 paragraphs; 100-char overlap preserves boundary context |
| **LangGraph over plain chains** | Graph structure makes multi-agent routing explicit and extensible |
| **Streamlit over Flask** | Rapid prototyping for data/AI apps, built-in state management |
| **OpenRouter for LLMs** | Single API key to access multiple free models, no vendor lock-in |
| **Service layer pattern** | Decouples UI (pages/) from business logic (services/) |

---

## PART 9 — Expected Interview Questions and Model Answers

### Q1: "Explain your project in 2 minutes."
> "StudyMate AI is an AI-powered study assistant I built using Python and Streamlit. The core idea: a student uploads a lecture PDF, and the app uses a RAG pipeline — Retrieval-Augmented Generation — to make the document searchable by an LLM. I built 4 AI features on top: document summarization, an auto-generated MCQ quiz, a question-answering chat, and a learning reflection module. I used LangGraph to build the multi-agent workflow, ChromaDB as a local vector database, and routed different tasks to specialized LLMs through OpenRouter."

---

### Q2: "What is RAG and why did you use it?"
> "RAG stands for Retrieval-Augmented Generation. The problem with sending a whole PDF to an LLM is token limits and cost — a 50-page document won't fit. RAG solves this by first converting the document into searchable vector embeddings stored in a database. When a user asks a question, we find only the top relevant chunks, inject them into the prompt, and let the LLM answer from that context. This keeps the answer grounded in actual document content rather than hallucinated memory."

---

### Q3: "What are embeddings and how do they work?"
> "An embedding is a numerical representation of text — a vector of floating-point numbers. The key property is semantic similarity: similar sentences produce similar vectors. I used the `all-MiniLM-L6-v2` model from SentenceTransformers, which converts any text into a 384-dimensional vector. When a user asks a question, that question gets embedded too, and ChromaDB finds the stored chunks whose vectors are closest to the question vector using cosine similarity."

---

### Q4: "What is LangGraph and why not just use LangChain?"
> "LangChain is great for single LLM calls with chains. LangGraph extends it for multi-step, stateful workflows. Think of it as a directed graph where each node is an agent performing a specific task. In my project, the graph has 4 nodes: route, retrieve, explain, and flashcard. After retrieval, the graph branches based on intent. LangGraph makes this branching logic explicit, reusable, and easy to extend without spaghetti code."

---

### Q5: "What is ChromaDB?"
> "ChromaDB is an open-source vector database optimized for storing and querying embeddings. I chose it because it runs locally with no server setup — just a folder on disk. It supports persistent storage, meaning the embeddings survive between app restarts. The API is simple: add documents with their embeddings, then query with a new embedding to get the top-N most similar ones."

---

### Q6: "How did you handle the two-agent architecture?"
> "I built an `AIProvider` class that acts as a model router. It has a task-to-model mapping: summary and quiz tasks go to Model 1 (which needs strong instruction-following for structured output), while question-answering and reflection go to Model 2 (which needs deeper reasoning). Any service just calls `get_llm(task='quiz')` and the provider returns the right LLM. This decouples the model selection from the business logic."

---

### Q7: "What challenges did you face?"
> "Three main ones. First, a NumPy 2.0 incompatibility broke ChromaDB on startup — solved by upgrading to ChromaDB 1.5.9. Second, free LLMs on OpenRouter were unreliable — Nemotron Lightning kept timing out with 504 errors, and Apodex was returning completely empty responses. I tested all available free models programmatically, found only `nvidia/nemotron-3-super-120b-a12b:free` worked reliably, and updated the routing accordingly. Third, some reasoning models put their output in `reasoning_content` instead of the standard `content` field, so I added a fallback to check both."

---

### Q8: "How would you improve this project?"
> "Several ways: add support for multiple simultaneous documents; implement user authentication so each user has their own document store; add streaming responses so answers appear word-by-word; swap the local embedding model for a cloud one like OpenAI embeddings for better accuracy; and add evaluation metrics — BLEU score or LLM-as-judge — to measure answer quality objectively."

---

### Q9: "What is cosine similarity?"
> "It measures the angle between two vectors, not their magnitude. A cosine similarity of 1 means the vectors point in exactly the same direction — the texts are semantically identical. 0 means unrelated. ChromaDB uses this to find which stored document chunks are most semantically similar to the user's query, so even if exact words don't match, related concepts are still retrieved."

---

### Q10: "What is chunk overlap and why does it matter?"
> "When you split a document into fixed-size chunks, important information can land exactly at a boundary — the first half of a sentence in one chunk, the second in the next. Overlap solves this: my chunks are 500 chars with 100 chars of overlap, meaning the last 100 characters of each chunk are repeated at the start of the next. This ensures boundary-spanning concepts are fully represented in at least one chunk."

---

## PART 10 — Numbers to Remember

| Metric | Value |
|---|---|
| Chunk size | 500 characters |
| Chunk overlap | 100 characters |
| Embedding dimensions | 384 (all-MiniLM-L6-v2) |
| Top-K retrieval | 8 chunks |
| Summary context cap | 15,000 characters |
| Default quiz questions | 5 MCQs |
| LLM max tokens | 8,192 |
| LLM timeout | 60 seconds |

---

## PART 11 — Quick Vocab Cheatsheet

| Term | Simple Definition |
|---|---|
| **RAG** | "Look it up first, then answer" pattern for LLMs |
| **Embedding** | Text converted to numbers that capture meaning |
| **Vector DB** | Database optimized for finding similar embeddings |
| **Chunk** | A small piece of the original document |
| **LLM** | Large Language Model — the AI that generates text |
| **Agent** | An LLM given a specific role and context |
| **LangGraph** | Framework for building multi-step AI workflows as graphs |
| **OpenRouter** | API gateway that lets you use many LLMs with one API key |
| **Session State** | Streamlit's way of remembering data between user interactions |
| **Cosine Similarity** | How similar two vectors are (1 = identical, 0 = unrelated) |

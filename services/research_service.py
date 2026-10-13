"""Self-reflective RAG research agent.

Flow: retrieve -> draft (notes only) -> web search -> validate -> revise.

Each step is a public method so the UI can render the agent's progress
live, and `run()` chains them for single-call / testing use.
"""

import json
import re

from rag.vector_store import VectorStore
from agents.ai_provider import AIProvider
from config import TAVILY_API_KEY, TAVILY_MAX_RESULTS, RETRIEVAL_TOP_K


def _response_text(resp):
    """Extract text from an LLM response, handling reasoning models."""
    text = (getattr(resp, "content", None) or "").strip()
    if not text:
        text = ((getattr(resp, "additional_kwargs", None) or {}).get("reasoning_content") or "").strip()
    return text


def _strip_code_fences(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"\s*```$", "", text, flags=re.DOTALL)
    return text.strip()


class ResearchService:
    """Generate -> web-verify -> self-correct for grounded answers."""

    def __init__(self):
        self.vector_store = VectorStore()
        self.ai_provider = AIProvider()

    def _llm(self):
        return self.ai_provider.get_llm(task="question_answering")

    # ------------------------------------------------------------
    # Step 1: retrieve
    # ------------------------------------------------------------
    def retrieve(self, question):
        results = self.vector_store.search(question, RETRIEVAL_TOP_K)
        documents = results.get("documents", [])
        context = "\n\n---\n\n".join(documents)
        return {"context": context, "documents": documents}

    # ------------------------------------------------------------
    # Step 2: draft (notes only)
    # ------------------------------------------------------------
    def generate_draft(self, question, context):
        prompt = f"""You are StudyMate AI, an expert tutor.
Answer the question using ONLY the lecture notes below.
Do not use outside knowledge. Be clear and exam-focused.

LECTURE NOTES:
{context}

QUESTION:
{question}

Answer:"""
        try:
            resp = self._llm().invoke(prompt)
            return _response_text(resp) or "The model returned an empty draft."
        except Exception as e:
            return f"[Draft generation failed: {e}]"

    # ------------------------------------------------------------
    # Step 3: web search (Tavily)
    # ------------------------------------------------------------
    def web_search(self, question):
        if not TAVILY_API_KEY:
            return {"sources": [], "error": "Tavily API key is not set (add TAVILY_API_KEY to .env)."}
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=TAVILY_API_KEY)
            resp = client.search(
                query=question,
                search_depth="advanced",
                max_results=TAVILY_MAX_RESULTS,
                include_answer=False,
            )
            raw = resp.get("results", []) if isinstance(resp, dict) else (getattr(resp, "results", None) or [])
            sources = []
            for r in raw:
                if isinstance(r, dict):
                    sources.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": (r.get("content") or "")[:1500],
                    })
                else:
                    sources.append({
                        "title": getattr(r, "title", ""),
                        "url": getattr(r, "url", ""),
                        "content": (getattr(r, "content", "") or "")[:1500],
                    })
            return {"sources": sources, "error": None}
        except Exception as e:
            return {"sources": [], "error": str(e)}

    # ------------------------------------------------------------
    # Step 4: validate draft against web
    # ------------------------------------------------------------
    def validate(self, question, draft, web):
        sources = (web or {}).get("sources", [])
        if not sources:
            return {
                "verdict": "skipped",
                "summary": (web or {}).get("error") or "No web results to validate against.",
                "checks": [],
            }
        web_text = "\n\n".join(
            f"[Source {i + 1}] {s.get('title', '')} ({s.get('url', '')})\n{s.get('content', '')}"
            for i, s in enumerate(sources)
        )
        prompt = f"""You are a rigorous fact-checker for a study assistant.
Compare the DRAFT answer (written only from a student's lecture notes)
against the WEB SEARCH RESULTS.

For each important claim in the draft, assign one status:
- "validated": the web results agree / confirm it.
- "corrected": the draft is wrong, outdated, or differs from the web; put the correction in "note".
- "added": an important point present in the web but MISSING from the draft.

Return ONLY valid JSON, no markdown:
{{
  "verdict": "needs_revision" or "accurate",
  "summary": "short one-line summary",
  "checks": [
    {{"claim": "...", "status": "validated|corrected|added", "note": "..."}}
  ]
}}

DRAFT ANSWER:
{draft}

WEB SEARCH RESULTS:
{web_text}"""
        try:
            resp = self._llm().invoke(prompt)
            content = _strip_code_fences(_response_text(resp))
            data = json.loads(content)
            return {
                "verdict": data.get("verdict", "needs_revision"),
                "summary": data.get("summary", ""),
                "checks": data.get("checks", []),
            }
        except Exception as e:
            return {"verdict": "unknown", "summary": f"Validation unavailable ({e})", "checks": []}

    # ------------------------------------------------------------
    # Step 5: revise
    # ------------------------------------------------------------
    def revise(self, question, context, draft, validation):
        validation_json = json.dumps(validation, ensure_ascii=False, default=str)
        prompt = f"""You are StudyMate AI.
Rewrite the DRAFT answer into a final, corrected answer for the student.
Incorporate every "corrected" and "added" point from the VALIDATION REPORT.
Do not invent anything beyond the lecture notes and the validated web facts.
Where the web corrected the notes, mention it briefly.
Keep it clear, exam-focused, with headings/bullets where helpful.
At the end, add a short section "### What I corrected after web-checking" listing the changes.

DRAFT ANSWER:
{draft}

VALIDATION REPORT (JSON):
{validation_json}

LECTURE NOTES (context):
{context}

QUESTION:
{question}

Final answer:"""
        try:
            resp = self._llm().invoke(prompt)
            return _response_text(resp) or "The model returned an empty revision."
        except Exception as e:
            return draft + f"\n\n_[Revision failed: {e}]_"

    # ------------------------------------------------------------
    # Full run (single call / tests)
    # ------------------------------------------------------------
    def run(self, question):
        retrieval = self.retrieve(question)
        draft = self.generate_draft(question, retrieval["context"])
        web = self.web_search(question)
        validation = self.validate(question, draft, web)
        final = self.revise(question, retrieval["context"], draft, validation)
        return {
            "retrieval": retrieval,
            "draft": draft,
            "web": web,
            "validation": validation,
            "final": final,
        }

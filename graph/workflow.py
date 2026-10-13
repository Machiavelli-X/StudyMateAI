import json
import re
from langgraph.graph import StateGraph, START, END
from graph.state import StudyState
from agents.ai_provider import AIProvider
from rag.vector_store import VectorStore
from config import RETRIEVAL_TOP_K


class StudyMateGraph:
    def __init__(self):
        provider = AIProvider()
        self.llm_explain = provider.get_apodex_llm()
        self.llm_flashcard = provider.get_nemotron_llm()
        self.store = VectorStore()
        self.graph = self._build()

    def _build(self):
        builder = StateGraph(StudyState)
        builder.add_node("route", self.route_query)
        builder.add_node("retrieve", self.retrieve)
        builder.add_node("explain", self.explain_agent)
        builder.add_node("flashcard", self.flashcard_agent)

        builder.add_edge(START, "route")
        builder.add_edge("route", "retrieve")
        builder.add_conditional_edges(
            "retrieve",
            lambda state: state.get("intent", "explain"),
            {"explain": "explain", "flashcard": "flashcard"},
        )
        builder.add_edge("explain", END)
        builder.add_edge("flashcard", END)
        return builder.compile()

    def route_query(self, state: StudyState):
        q = state["question"].lower()
        flashcard_words = ["flashcard", "cards", "anki", "memorize", "deck"]
        intent = "flashcard" if any(w in q for w in flashcard_words) else "explain"
        return {"intent": intent}

    def retrieve(self, state: StudyState):
        results = self.store.search(state["question"], RETRIEVAL_TOP_K)
        context = "\n\n---\n\n".join(r["text"] for r in results)
        return {"context": context, "sources": results}

    def explain_agent(self, state: StudyState):
        prompt = f"""You are an Expert Teacher.
Use ONLY the retrieved study material to explain the concept.
Explain it as if you are talking to a 5-year-old, using a fun and simple analogy.
If the material does not contain the answer, say exactly that you could not find it.

RETRIEVED STUDY MATERIAL:
{state.get('context', '')}

STUDENT QUESTION:
{state['question']}
"""
        response = self.llm_explain.invoke(prompt)
        return {"answer": response.content.strip()}

    def flashcard_agent(self, state: StudyState):
        prompt = f"""You are a Memorization Assistant.
Generate Anki-style flashcards ONLY from the retrieved study material.
Return ONLY a JSON array. No Markdown and no extra text.
Generate 5 flashcards. Each object must contain:
front (the question/concept), back (the short answer/definition).

RETRIEVED STUDY MATERIAL:
{state.get('context', '')}

STUDENT REQUEST:
{state['question']}
"""
        response = self.llm_flashcard.invoke(prompt)
        content = response.content.strip()
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.DOTALL).strip()
        cards = json.loads(content)
        return {"answer": json.dumps(cards, ensure_ascii=False)}

    def run(self, question: str):
        return self.graph.invoke({"question": question})

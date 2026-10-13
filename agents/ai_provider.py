import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from config import (
    GROQ_API_KEY,
    OPENROUTER_API_KEY,
    APODEX_API_KEY,
    GROQ_MODEL,
    OPENROUTER_MODEL,
    APODEX_MODEL,
    NEMOTRON_MODEL,
    OPENROUTER_BASE_URL,
    APODEX_BASE_URL,
)

load_dotenv()


class AIProvider:
    """
    Manages multi-model task-based selection between Model 1 (Apodex/Groq)
    and Model 2 (Nemotron/OpenRouter).
    """

    TASK_PROVIDER_MAP = {
        "summary": "model_1",
        "quiz": "model_1",
        "question_answering": "model_2",
        "reflection": "model_2",
    }

    def __init__(self):
        self.groq_api_key = GROQ_API_KEY
        self.openrouter_api_key = OPENROUTER_API_KEY
        self.apodex_api_key = APODEX_API_KEY

        self.groq_model = GROQ_MODEL
        self.openrouter_model = OPENROUTER_MODEL
        self.apodex_model = APODEX_MODEL
        self.nemotron_model = NEMOTRON_MODEL

    def _get_model_1(self) -> ChatOpenAI:
        """Model 1: Apodex (or Groq if configured). Used for Summary & Quiz."""
        # 1. If direct APODEX key is available
        if self.apodex_api_key and "api.apodex.ai" in APODEX_BASE_URL:
            return ChatOpenAI(
                model=self.apodex_model,
                api_key=self.apodex_api_key,
                base_url=APODEX_BASE_URL,
                temperature=0.2,
                max_tokens=8192,
                request_timeout=60,
            )

        # 2. If Groq is explicitly configured with a key
        if self.groq_api_key:
            try:
                from langchain_groq import ChatGroq
                return ChatGroq(
                    groq_api_key=self.groq_api_key,
                    model_name=self.groq_model,
                    temperature=0.3,
                )
            except Exception:
                pass

        # 3. Via OpenRouter (e.g. apodex/apodex-1.1-mini:free or other Model 1)
        if self.openrouter_api_key:
            return ChatOpenAI(
                model=self.apodex_model,
                api_key=self.openrouter_api_key,
                base_url=OPENROUTER_BASE_URL,
                temperature=0.2,
                max_tokens=8192,
                request_timeout=60,
            )

        raise ValueError("No API key found for Model 1 (Apodex/Groq/OpenRouter). Please check your .env file.")

    def _get_model_2(self) -> ChatOpenAI:
        """Model 2: Nemotron (or OpenRouter reasoning model). Used for Ask AI & Reflection."""
        if self.openrouter_api_key:
            return ChatOpenAI(
                model=self.nemotron_model,
                api_key=self.openrouter_api_key,
                base_url=OPENROUTER_BASE_URL,
                temperature=0.3,
                max_tokens=8192,
                request_timeout=60,
            )

        # Fallback to Model 1 if OpenRouter key is not set but Apodex is
        return self._get_model_1()

    def get_llm(self, provider="auto", task=None):
        """
        Return an LLM using task-based routing for provider='auto',
        or explicit provider selection.
        """
        provider_lower = (provider or "auto").lower()

        if provider_lower in ("groq", "apodex", "model_1"):
            return self._get_model_1()

        if provider_lower in ("openrouter", "nemotron", "model_2"):
            return self._get_model_2()

        if provider_lower == "auto":
            target = self.TASK_PROVIDER_MAP.get(task, "model_2")
            if target == "model_1":
                try:
                    return self._get_model_1()
                except Exception:
                    return self._get_model_2()
            else:
                try:
                    return self._get_model_2()
                except Exception:
                    return self._get_model_1()

        # Default fallback
        return self._get_model_2()

    def available_providers(self):
        providers = []
        if self.groq_api_key:
            providers.append("Groq")
        if self.apodex_api_key:
            providers.append("Apodex")
        if self.openrouter_api_key:
            providers.append("OpenRouter (Apodex & Nemotron)")
        return providers

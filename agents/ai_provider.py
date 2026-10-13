from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from config import (
    GROQ_API_KEY,
    OPENROUTER_API_KEY,
    APODEX_API_KEY,
    LLM_MODEL,
    OPENROUTER_BASE_URL,
    APODEX_BASE_URL,
)

load_dotenv()


class AIProvider:
    """
    Returns a single LLM for every feature.

    Provider fallback order: OpenRouter -> Groq -> Apodex,
    all using the one LLM_MODEL from config.
    """

    def __init__(self):
        self.groq_api_key = GROQ_API_KEY
        self.openrouter_api_key = OPENROUTER_API_KEY
        self.apodex_api_key = APODEX_API_KEY
        self.model = LLM_MODEL

    def _get_openrouter(self):
        return ChatOpenAI(
            model=self.model,
            api_key=self.openrouter_api_key,
            base_url=OPENROUTER_BASE_URL,
            temperature=0.3,
            max_tokens=8192,
            request_timeout=60,
        )

    def _get_groq(self):
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                groq_api_key=self.groq_api_key,
                model_name=self.model,
                temperature=0.3,
            )
        except Exception:
            return None

    def _get_apodex(self):
        return ChatOpenAI(
            model=self.model,
            api_key=self.apodex_api_key,
            base_url=APODEX_BASE_URL,
            temperature=0.2,
            max_tokens=8192,
            request_timeout=60,
        )

    def get_llm(self, provider="auto", task=None):
        """
        Return the single LLM.

        `task` is kept for backward compatibility but no longer changes the
        model — every feature uses the same LLM_MODEL.
        """
        provider_lower = (provider or "auto").lower()

        if provider_lower == "groq" and self.groq_api_key:
            llm = self._get_groq()
            if llm is not None:
                return llm

        if provider_lower == "apodex" and self.apodex_api_key:
            return self._get_apodex()

        if provider_lower == "openrouter" and self.openrouter_api_key:
            return self._get_openrouter()

        # Default / auto: OpenRouter first, then Groq, then Apodex
        if self.openrouter_api_key:
            return self._get_openrouter()
        if self.groq_api_key:
            llm = self._get_groq()
            if llm is not None:
                return llm
        if self.apodex_api_key:
            return self._get_apodex()

        raise ValueError("No API key found. Please add OPENROUTER_API_KEY to your .env file.")

    def available_providers(self):
        providers = []
        if self.openrouter_api_key:
            providers.append("OpenRouter")
        if self.groq_api_key:
            providers.append("Groq")
        if self.apodex_api_key:
            providers.append("Apodex")
        return providers

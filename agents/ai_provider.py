from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    GROQ_API_KEY,
    APODEX_API_KEY,
    APODEX_BASE_URL,
)

load_dotenv()


class AIProvider:
    """
    Returns a single LLM (any OpenAI-compatible endpoint).

    Primary provider: LLM_API_KEY + LLM_BASE_URL (e.g. OpenCode Zen, OpenRouter).
    Optional fallbacks: Groq, Apodex.
    """

    def __init__(self):
        self.api_key = LLM_API_KEY
        self.base_url = LLM_BASE_URL
        self.model = LLM_MODEL
        self.groq_api_key = GROQ_API_KEY
        self.apodex_api_key = APODEX_API_KEY

    def _get_llm(self):
        return ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=0.3,
            max_tokens=8192,
            request_timeout=60,
            # OpenCode Zen "go" requires a stable session id per conversation.
            default_headers={"x-opencode-session": "studymate-session"},
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
        Return the single LLM. `task` is kept for backward compatibility
        but no longer changes the model.
        """
        provider_lower = (provider or "auto").lower()

        if provider_lower == "groq" and self.groq_api_key:
            llm = self._get_groq()
            if llm is not None:
                return llm

        if provider_lower == "apodex" and self.apodex_api_key:
            return self._get_apodex()

        # Default / auto: the primary LLM provider
        if self.api_key:
            return self._get_llm()
        if self.groq_api_key:
            llm = self._get_groq()
            if llm is not None:
                return llm
        if self.apodex_api_key:
            return self._get_apodex()

        raise ValueError("No API key found. Please add LLM_API_KEY (and LLM_BASE_URL) to your .env file.")

    def available_providers(self):
        providers = []
        if self.api_key:
            providers.append("LLM")
        if self.groq_api_key:
            providers.append("Groq")
        if self.apodex_api_key:
            providers.append("Apodex")
        return providers

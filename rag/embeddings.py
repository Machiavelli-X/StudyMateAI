import streamlit as st
from sentence_transformers import SentenceTransformer


@st.cache_resource
def _load_model():
    """Load the embedding model once and reuse it across all pages and reruns."""
    return SentenceTransformer("all-MiniLM-L6-v2")


class EmbeddingModel:
    """
    Generates vector embeddings
    for text chunks.
    """

    def __init__(self):
        self.model = _load_model()

    def encode(self, texts):
        return self.model.encode(texts)
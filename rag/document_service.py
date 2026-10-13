from pathlib import Path
import fitz
from rag.chunker import TextChunker
from rag.vector_store import VectorStore
from config import UPLOAD_FOLDER


class DocumentService:
    def __init__(self):
        self.chunker = TextChunker()
        self.vector_store = VectorStore()

    def ingest_pdf(self, uploaded_file):
        Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
        path = Path(UPLOAD_FOLDER) / uploaded_file.name
        path.write_bytes(uploaded_file.getbuffer())

        pages = []
        with fitz.open(path) as doc:
            for page_number, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if text:
                    pages.append(f"[Page {page_number}]\n{text}")

        full_text = "\n\n".join(pages)
        if not full_text:
            raise ValueError("No readable text was found in this PDF.")

        chunks = self.chunker.split_text(full_text)
        self.vector_store.clear()
        count = self.vector_store.add_documents(chunks)
        return {"filename": uploaded_file.name, "pages": len(pages), "chunks": count}

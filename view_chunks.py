import sys
import chromadb
from config import CHROMA_DB_PATH, COLLECTION_NAME

# Ensure UTF-8 output in Windows PowerShell/cmd terminals
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def view_all_chunks():
    print(f"Connecting to ChromaDB at: {CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"Collection '{COLLECTION_NAME}' not found or empty: {e}")
        return

    data = collection.get()
    ids = data.get("ids", [])
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", []) or [{}] * len(ids)

    total = len(documents)
    print(f"\nFound {total} chunks stored in '{COLLECTION_NAME}' collection.\n" + "=" * 60)

    if total == 0:
        print("No chunks stored yet. Upload a PDF first via the web app.")
        return

    for i, (chunk_id, doc) in enumerate(zip(ids, documents), start=1):
        print(f"\n--- [Chunk #{i} | ID: {chunk_id}] ---")
        print(doc.strip())
        print("-" * 60)

    print(f"\nTotal Chunks: {total}")


if __name__ == "__main__":
    view_all_chunks()

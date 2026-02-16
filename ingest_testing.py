from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

PERSIST_DIR = "rules_rag/chroma_db"
COLLECTION = "mtg_rules_and_cards"


def get_or_build_vectorstore(docs, embeddings):
    """
    If a persisted Chroma DB exists and has vectors, load it.
    Otherwise, build it (split + embed) and persist.
    """

    # Always try to load first
    db = Chroma(
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION,
        embedding_function=embeddings,
    )

    # If it already has vectors, you're done (no re-embed)
    try:
        existing = db._collection.count()
    except Exception:
        existing = 0

    if existing and existing > 0:
        print(f"Loaded existing Chroma DB ({existing} vectors).")
        return db

    # Otherwise build it
    print("No existing vectors found. Building Chroma DB (this will embed once)...")

    """splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150)
    splits = splitter.split_documents(docs)"""

    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION,
    )

    # Some versions persist automatically; harmless if called anyway
    try:
        db.persist()
    except Exception:
        pass

    print(f"Built and persisted Chroma DB ({len(docs)} chunks).")
    return db

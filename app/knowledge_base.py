from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
import os
import hashlib

# langchain==0.2.17
# langchain-cli==0.0.30
# langchain-community==0.2.19
# langchain-core==0.2.43
# langchain-ollama==0.1.3
# langchain-text-splitters==0.2.4

def file_hash(path):
    """Create a simple hash based on path and last modified time."""
    stat = os.stat(path)
    return hashlib.md5(f"{path}-{stat.st_mtime}".encode()).hexdigest()

def load_new_documents(folder_path, existing_hashes):
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt") or filename.endswith(".md"):
            filepath = os.path.join(folder_path, filename)
            fhash = file_hash(filepath)
            if fhash in existing_hashes:
                continue  # skip already processed file
            loader = TextLoader(filepath)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_docs = splitter.split_documents(docs)
            # Add metadata with the hash so we can check it next time
            for doc in split_docs:
                doc.metadata["file_hash"] = fhash
            all_docs.extend(split_docs)
    return all_docs

def build_vectorstore(folder_path="docs"):
    embed = OllamaEmbeddings(model="nomic-embed-text", base_url="http://192.168.55.1:11434")

    # Load existing vectorstore or create new one
    vector_store = Chroma(
        embedding_function=embed,
        persist_directory="chroma_db",
        collection_name="my_collection"
    )

    # Extract all existing file hashes
    existing_docs = vector_store.get(include=["metadatas"])
    existing_hashes = set(
        metadata["file_hash"]
        for metadata in existing_docs["metadatas"]
        if "file_hash" in metadata
    )

    print(f"[DEBUG] Skipping {len(existing_hashes)} already indexed files...")

    new_documents = load_new_documents(folder_path, existing_hashes)

    if new_documents:
        vector_store.add_documents(new_documents)
        print(f"[DEBUG] Added {len(new_documents)} new documents.")
    else:
        print("[DEBUG] No new documents to add.")

    doc_count = vector_store._collection.count()
    print(f"[DEBUG] Vector store contains {doc_count} documents.")

if __name__ == "__main__":
    build_vectorstore()
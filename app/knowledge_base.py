from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os
import hashlib


# langchain                              0.3.26
# langchain-chroma                       0.2.4
# langchain-community                    0.3.26
# langchain-core                         0.3.67
# langchain-huggingface                  0.3.0
# langchain-ollama                       0.3.3
# langchain-text-splitters               0.3.8


def file_hash(path):
    # Create a simple hash based on path and last modified time
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
    embed = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )

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

    print(f"[DEBUG] Skipping {len(existing_hashes)} already indexed files...", flush=True)

    new_documents = load_new_documents(folder_path, existing_hashes)

    print("1", flush=True)

    if new_documents:
        print("2", flush=True)
        vector_store.add_documents(new_documents)
        print(f"[DEBUG] Added {len(new_documents)} new documents.", flush=True)
    else:
        print("[DEBUG] No new documents to add.", flush=True)

    doc_count = vector_store._collection.count()
    print(f"[DEBUG] Vector store contains {doc_count} documents.", flush=True)

if __name__ == "__main__":
    build_vectorstore()
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
import os

def load_documents_from_folder(folder_path):
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt") or filename.endswith(".md"):
            filepath = os.path.join(folder_path, filename)
            loader = TextLoader(filepath)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_docs = splitter.split_documents(docs)
            all_docs.extend(split_docs)
    return all_docs

def build_vectorstore(folder_path="docs"):
    embed = OllamaEmbeddings(model="tinyllama", base_url="http://localhost:11434")
    documents = load_documents_from_folder(folder_path)

    vector_store = Chroma.from_documents(
        documents,
        embedding=embed,
        persist_directory="chroma_db",
        collection_name="my_collection"
    )

    # Re-initialize to load the persisted store
    persisted_store = Chroma(
        collection_name="my_collection",
        persist_directory="chroma_db",
        embedding_function=embed
    )

    doc_count = persisted_store._collection.count()
    print(f"[DEBUG] Vector store contains {doc_count} documents.")
if __name__ == "__main__":
    build_vectorstore()

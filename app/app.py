from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
import streamlit as st
import os

def query_llm(prompt, model_name="tinyllama"):
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    llm = OllamaLLM(model=model_name, base_url=ollama_host)
    embed = OllamaEmbeddings(model="tinyllama", base_url=ollama_host)

    vector_store = Chroma(
        collection_name="my_collection",
        persist_directory="chroma_db",
        embedding_function=embed
    )

    print(f"[DEBUG] Total documents in vectorstore: {vector_store._collection.count()}", flush=True)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # `k=3` gets top 3 matches
    docs = retriever.invoke(prompt)

    print(f"[DEBUG] Retrieved {len(docs)} documents:", flush=True)
    for d in docs:
        print(d.page_content[:200], flush=True)  # First 200 chars of each result

    qa_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, qa_chain)

    return chain.invoke({"input": prompt})


def main():
    st.title("Torch Technologies Chatbot")

    # Ollama model selection
    ollama_model = st.selectbox("Select Ollama Model:", options=["tinyllama", "mistral", "llama2", "orca-mini", "phi"], index=0)

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User input
    user_query = st.text_input("Ask a question:")

    if user_query:
        try:
            with st.spinner("Thinking..."):
                response = query_llm(user_query, model_name=ollama_model)  # Query the selected model
                st.session_state.chat_history.append({"user": user_query, "llm": response})
        except Exception as e:
            st.error(f"Error querying model: {e}")

    # Display chat history
    st.header("Chat History")
    for msg in reversed(st.session_state.chat_history):
        st.write(f"**User:** {msg['user']}")
        st.write(f"**LLM:** {msg['llm']}")
        st.divider()

if __name__ == "__main__":
    main()

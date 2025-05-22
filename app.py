from langchain_ollama import OllamaLLM
import streamlit as st
import os
import time


def query_llm(prompt, model_name="mistral"):
    ollama_host = "http://10.3.11.84:11434"
    llm = OllamaLLM(model=model_name, base_url=ollama_host)

    start = time.time()
    #print(f"LLM start time: {start:.2f} seconds")

    result = llm.invoke(prompt)
    end = time.time()

    #print(f"LLM end time: {end:.2f} seconds")
    return result


def main():
    st.title("Torch Technologies Chatbot")

    # Ollama model selection
    ollama_model = st.selectbox("Select Ollama Model:", options=["mistral", "llama2", "orca-mini", "phi"], index=0)

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
    for msg in st.session_state.chat_history:
        st.write(f"**User:** {msg['user']}")
        st.write(f"**LLM:** {msg['llm']}")
        st.divider()

if __name__ == "__main__":
    main()
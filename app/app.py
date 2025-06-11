from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
import streamlit as st
import os


# langchain==0.2.17
# langchain-cli==0.0.30
# langchain-community==0.2.19
# langchain-core==0.2.43
# langchain-ollama==0.1.3
# langchain-text-splitters==0.2.4


def generate_suggestions(chat_history, model_name="gemma:7b"):
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    llm = OllamaLLM(model=model_name, base_url=ollama_host)

    history_text = "\n".join(f"User: {m['user']}\nAI: {m['llm']}" for m in chat_history[-3:])  # Last 3 turns

    suggestion_prompt = f"""
    You are a helpful assistant.

    Given the following conversation between a user and an AI, suggest 3 thoughtful follow-up questions the user might ask next. Keep the suggestions relevant and curious.

    Conversation:
    {history_text}

    Suggestions:
    """

    try:
        suggestions = llm.invoke(suggestion_prompt)
        print("Suggested questions:", suggestions)
        lines = [line.strip("-• \n") for line in suggestions.splitlines() if line.strip()]
        return lines[:3]  # Return up to 3 suggestions
    except Exception as e:
        print(f"Suggestion generation failed: {e}")
        return []


def query_llm(prompt, chat_history=None, model_name="gemma:7b"):
    if chat_history is None:
        chat_history = []

    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    llm = OllamaLLM(model=model_name, base_url=ollama_host)
    embed = OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_host)

    vector_store = Chroma(
        collection_name="my_collection",
        persist_directory="chroma_db",
        embedding_function=embed
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    history_context = "\n".join(
        [f"User: {msg['user']}\nAI: {msg['llm']}" for msg in chat_history]
    )

    # Basic prompt template
    template = """You are a friendly assistant. Answer the question based on the context provided below.

    If the answer is not in the context, say "I don't know."

    Context:
    {context}

    Conversation so far:
    {chat_history}
    
    Now, answer this question:
    {input}
    """
    prompt_template = PromptTemplate.from_template(template)

    qa_chain = create_stuff_documents_chain(llm, prompt_template)
    chain = create_retrieval_chain(retriever, qa_chain)

    output = chain.invoke({
        "input": prompt,
        "chat_history": history_context,
    })

    return output['answer']


def main():
    st.title("Torch Technologies Chatbot")
    ollama_model = "gemma:7b"

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "recommended" not in st.session_state:
        st.session_state.recommended = []

    selected_question = None

    # # Display recommended questions
    # if st.session_state.recommended:
    #     st.subheader("Recommended Questions")
    #     selected_question = st.radio(
    #         "Pick a suggested question:", 
    #         options=st.session_state.recommended,
    #         key="question_suggestion"
    #     )

    # Capture user input
    if selected_question:
        user_query = st.text_input("Ask a question:", value=selected_question)
    else:
        user_query = st.text_input("Ask a question:", value="")

    # Only proceed if new question was asked
    if user_query.strip():
        with st.spinner("Thinking..."):
            try:
                response = query_llm(user_query, model_name=ollama_model, chat_history=st.session_state.chat_history)

                st.session_state.chat_history.append({
                    "user": user_query,
                    "llm": response,
                })

                st.session_state.recommended = generate_suggestions(
                    st.session_state.chat_history, model_name=ollama_model
                )

            except Exception as e:
                st.error(f"Error querying model: {e}")
                st.stop()

    # Display chat history
    st.header("Chat History")
    for msg in reversed(st.session_state.chat_history):
        st.write(f"**User:** {msg['user']}")
        st.write(f"**LLM:** {msg['llm']}")
        st.divider()


if __name__ == "__main__":
    main()

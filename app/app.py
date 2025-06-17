from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
import os
import re


# langchain==0.2.17
# langchain-cli==0.0.30
# langchain-community==0.2.19
# langchain-core==0.2.43
# langchain-ollama==0.1.3
# langchain-text-splitters==0.2.4

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = "llama3:8b"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

@st.cache_resource
def get_llm():
    return OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_HOST, streaming=True)

@st.cache_resource
def get_vector_store():
    embed = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )
    return Chroma(
        collection_name="my_collection",
        persist_directory="chroma_db",
        embedding_function=embed,
    )

@st.cache_resource
def get_prompt_template():
    template = """You are a friendly assistant, and an expert on Torch Technologies. Answer the questions to the best of your ability.
    Pretend you know all the answers yourself, and do not need any context to find the answers.

    Context:
    {context}

    Conversation so far:
    {chat_history}

    Now, answer this question:
    {input}
    """
    return PromptTemplate.from_template(template)


def generate_questions(chat_history):
    if chat_history == []:
        return []

    history_text = "\n".join(f"User: {m['user']}\nAI: {m['llm']}" for m in chat_history[-3:])  # Last 3 turns

    prompt = f"""
    You are a helpful assistant.

    Given the following conversation between a user and an AI, suggest 3 short follow-up questions the user might ask next. Keep the questions relevant but simple.

    Conversation:
    {history_text}

    Suggestions:
    """

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        print(response, flush=True)
        questions = re.findall(r"^\s*\d\.\s+(.*)", response, re.MULTILINE)
        return questions

    except Exception as e:
        print(f"Error generating questions: {e}")
        return []


def query_llm_stream(prompt, chat_history):
    history_context = "\n".join(
        [f"User: {msg['user']}\nAI: {msg['llm']}" for msg in chat_history]
    )
    llm = get_llm()
    vector_store = get_vector_store()
    prompt_template = get_prompt_template()
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    results = retriever.get_relevant_documents(prompt)
    for doc in results:
        print(doc.page_content, flush=True)

    qa_chain = create_stuff_documents_chain(llm, prompt_template)
    retrieval_chain = create_retrieval_chain(retriever, qa_chain)

    for chunk in retrieval_chain.stream({
        "input": prompt,
        "chat_history": history_context,
    }):
        yield chunk.get("answer", "")


def main():
    st.title("Torch Technologies Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    def set_question(question):
        st.session_state["my_question"] = question

    user_input = st.chat_input("Ask a question")

    # Display all previous messages
    for msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(msg["user"])
        with st.chat_message("assistant"):
            st.markdown(msg["llm"])

    # Determine current question
    if st.session_state.get("my_question"):
        question = st.session_state["my_question"]
        st.session_state["my_question"] = ""
    elif user_input:
        question = user_input
    else:
        question = None

    # Handle new question
    if question:
        with st.chat_message("user"):
            st.markdown(question)

        try:
            with st.chat_message("assistant"):
                msg_placeholder = st.empty()
                full_response = ""

                for chunk in query_llm_stream(question, chat_history=st.session_state.chat_history):
                    full_response += chunk
                    msg_placeholder.markdown(full_response + "▌")

                msg_placeholder.markdown(full_response)

            # Save to history
            st.session_state.chat_history.append({
                "user": question,
                "llm": full_response,
            })

            # Suggest follow-up questions
            followup_questions = generate_questions(st.session_state.chat_history)
            if followup_questions:
                with st.chat_message("assistant"):
                    followup_container = st.container()
                    for q in followup_questions:
                        if followup_container.button(q, on_click=set_question, args=(q,)):
                            followup_container.empty()
                            break

        except Exception as e:
            st.error(f"Error querying model: {e}")

if __name__ == "__main__":
    main()

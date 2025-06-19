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
FOLLOWUP_MODEL = "mistral"

@st.cache_resource
def get_llm():
    return OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_HOST, streaming=True)


@st.cache_resource
def get_fast_llm():
    return OllamaLLM(model=FOLLOWUP_MODEL, base_url=OLLAMA_HOST, temperature=0.7)


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
    template = """You are an expert on Torch Technologies. Answer the questions to the best of your ability.
    Do not say what your answer is based on.

    Context:
    {context}

    Conversation so far:
    {chat_history}

    Now, answer this question:
    {input}
    """
    return PromptTemplate.from_template(template)


def generate_questions(previous_response):
    if not previous_response:
        return []

    prompt = f"""
    Suggest 3 short and simple follow-up questions based on this information. Do not provide the answers.

    Information:
    {previous_response}
    """

    try:
        llm = get_fast_llm()
        response = llm.invoke(prompt)
        print(response, flush=True)
        questions = re.findall(r"^\s*\d\.\s+(.*)", response, re.MULTILINE)
        return questions

    except Exception as e:
        print(f"Error generating questions: {e}")
        return []


def query_llm(vector_store, prompt, history_context):
    llm = get_llm()
    prompt_template = get_prompt_template()
    retriever = vector_store.as_retriever(
        search_type = "mmr",
        search_kwargs={"k": 10, "fetch_k": 20}
    )

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

    vector_store = get_vector_store()

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

    # Display starter questions
    if not question:
        starter_questions = [
            "What is Torch Technologies?",
            "Who are Torch Technologies' major clients?",
            "What kind of projects does Torch Technologies work on?"
        ]

        with st.chat_message("assistant"):
            starter_container = st.container()
            for q in starter_questions:
                if starter_container.button(q, on_click=set_question, args=(q,)):
                    starter_container.empty()
                    break

    # Handle new question
    if question:
        with st.chat_message("user"):
            st.markdown(question)

        try:
            with st.chat_message("assistant"):
                msg_placeholder = st.empty()
                full_response = ""

                history_context = "\n".join(
                    [f"User: {msg['user']}\nAI: {msg['llm']}" for msg in st.session_state.chat_history[-3:]]
                )

                for chunk in query_llm(vector_store, question, history_context):
                    full_response += chunk
                    msg_placeholder.markdown(full_response + "▌")

                msg_placeholder.markdown(full_response)

            # Save to history
            st.session_state.chat_history.append({
                "user": question,
                "llm": full_response,
            })

            # Suggest follow-up questions
            followup_questions = generate_questions(full_response)
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

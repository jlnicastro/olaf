from langchain_ollama import OllamaLLM
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
import os
import re
from faster_whisper import WhisperModel
from tempfile import NamedTemporaryFile
from kokoro import KPipeline
import soundfile as sf
import io
import base64
import numpy as np
import hashlib


# langchain                              0.3.26
# langchain-chroma                       0.2.4
# langchain-community                    0.3.26
# langchain-core                         0.3.67
# langchain-huggingface                  0.3.0
# langchain-ollama                       0.3.3
# langchain-text-splitters               0.3.8


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = "mistral"
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
    template = """You are an expert on Torch Technologies. Answer any questions thoroughly.
    Do not say what your answer is based on.

    Context:
    {context}

    Conversation so far:
    {chat_history}

    Now, answer this question:
    {input}
    """
    return PromptTemplate.from_template(template)


@st.cache_resource
def load_whisper_model():
    return WhisperModel("small", compute_type="int8")


@st.cache_resource
def get_kokoro_pipeline():
    return KPipeline(lang_code="a")


def transcribe_audio():
    audio_bytes = st.session_state["uploaded_audio"].getvalue()
    with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio.flush()

        segments, _ = load_whisper_model().transcribe(
            temp_audio.name,
            language="en",
            beam_size=1
        )

        transcription = " ".join([seg.text for seg in segments])
        st.session_state["audio_input"] = transcription



def kokoro_generate(text, voice='af_heart'):
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")

    # Use a simple hash of the text + voice as the cache key
    key = hashlib.sha256((text + voice).encode()).hexdigest()

    if "tts_cache" not in st.session_state:
        st.session_state.tts_cache = {}

    # Return cached audio if available
    if key in st.session_state.tts_cache:
        return st.session_state.tts_cache[key]

    # Otherwise, generate new audio
    pipeline = get_kokoro_pipeline()
    generator = pipeline(text, voice=voice)

    all_audio = []
    for gs, ps, audio in generator:
        all_audio.append(audio)

    full_audio = np.concatenate(all_audio)

    wav_io = io.BytesIO()
    sf.write(wav_io, full_audio, 24000, format="WAV")
    wav_io.seek(0)
    audio_bytes = wav_io.read()

    # Cache the result
    st.session_state.tts_cache[key] = audio_bytes

    return audio_bytes


def generate_questions(previous_response):
    if not previous_response:
        return []

    prompt = f"""
    Based on the information below, suggest 3 short follow-up questions that someone knew to Torch Technologies might ask.
    Each question should be just one clause (no commas, conjunctions, or multi-part sentences).
    Avoid repeating anything already answered. Do not include the answers.
    Keep the follow-up questions relevant to Torch Technologies.

    Information:
    {previous_response}
    """

    try:
        llm = get_fast_llm()
        response = llm.invoke(prompt)
        print(response, flush=True)
        questions = re.findall(r"^\s*\d\.\s+['\"]?(.*?)['\"]?\s*$", response, re.MULTILINE)
        return questions

    except Exception as e:
        print(f"Error generating questions: {e}", flush=True)
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


def get_cached_response(question, history_context):
    # Create a unique hash key for question + context
    key = hashlib.sha256((question).encode()).hexdigest()

    if "llm_cache" not in st.session_state:
        st.session_state.llm_cache = {}

    # Return cached if exists
    if key in st.session_state.llm_cache:
        return st.session_state.llm_cache[key]

    # If not cached, run LLM and save result
    response = ""
    for chunk in query_llm(vector_store, question, history_context):
        response += chunk

    st.session_state.llm_cache[key] = response
    return response

######################################

st.title("Torch Technologies Chatbot")

st.sidebar.title("Settings")
use_tts = st.sidebar.checkbox("Enable Kokoro TTS", value=True)

vector_store = get_vector_store()
get_kokoro_pipeline()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def set_question(question):
    st.session_state["my_question"] = question

chat_input = st.chat_input("Ask a question")

# Display all previous messages
for msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(msg["user"])
    with st.chat_message("assistant", avatar='favicon-96x96.png'):
        st.markdown(msg["llm"])

# Determine current question
if st.session_state.get("my_question"):
    question = st.session_state["my_question"]
    st.session_state["my_question"] = ""
elif chat_input:
    question = chat_input
elif st.session_state.get("audio_input"):
    question = st.session_state.pop("audio_input")
else:
    question = None

# Display starter questions
if not question:
    starter_questions = [
        "What is Torch Technologies?",
        "Who are Torch Technologies' major clients?",
        "What kind of projects does Torch Technologies work on?"
    ]

    with st.chat_message("assistant", avatar='favicon-96x96.png'):
        starter_container = st.container()
        for q in starter_questions:
            if starter_container.button(q, on_click=set_question, args=(q,)):
                starter_container.empty()
                break

# Handle new question
if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar='favicon-96x96.png'):
        msg_placeholder = st.empty()
        full_response = ""

        history_context = "\n".join(
            [f"User: {msg['user']}\nAI: {msg['llm']}" for msg in st.session_state.chat_history[-3:]]
        )

        full_response = get_cached_response(question, history_context)
        msg_placeholder.markdown(full_response)

    if use_tts:
        final_audio_bytes = kokoro_generate(full_response)
        b64_audio = base64.b64encode(final_audio_bytes).decode('utf-8')

        audio_html = f"""
        <audio autoplay>
            <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
            Your browser does not support the audio element.
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)

    # Save to history
    st.session_state.chat_history.append({
        "user": question,
        "llm": full_response,
    })

    # Suggest follow-up questions
    followup_questions = generate_questions(full_response)
    if followup_questions:
        with st.chat_message("assistant", avatar='favicon-96x96.png'):
            followup_container = st.container()
            for q in followup_questions:
                if followup_container.button(q, on_click=set_question, args=(q,)):
                    followup_container.empty()
                    break

st.audio_input(
    "Or record a voice question",
    key="uploaded_audio",
    on_change=transcribe_audio
)
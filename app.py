import streamlit as st
import uuid
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
import warnings
warnings.filterwarnings('ignore')

from src.rag_components.get_chroma_db import get_chroma_db
from src.rag_components.get_embeddingfile import get_embedding




model = HuggingFaceHub(
    huggingfacehub_api_token="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",
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={'temperature': 1, "max_length": 180}
)


# Prompt setup
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use two sentences maximum and keep the answer concise.\n\n{context}"
)

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)


contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def get_session_id():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# give the title
st.title("Welcome to Gau")
session_id = get_session_id()
st.sidebar.title("Session Info")
st.sidebar.write(f"Session ID: {session_id}")

vectors = get_chroma_db()
retriever = vectors.as_retriever(search_kwargs={"k": 3})
history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

store = {}

# Conversational chain setup with session history
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# User input
user_input = st.text_input("Ask a question:", "")

# Submit the form and trigger the conversation
if st.button("Send"):
    if user_input:
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )["answer"].split("\n\n")[-1]

        st.write(f"Response: {response}")
    else:
        st.write("Please enter a question.")

import streamlit as st
from pathlib import Path
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

PDF_FOLDER = Path(__file__).parent / 'PDF'
MODEL_NAME = 'gemini-1.5-flash'


def document_loader():
    documents = []
    for pdf in PDF_FOLDER.glob('*.pdf'):
        loader = PyPDFLoader(str(pdf))
        pdf_documents = loader.load()
        documents.extend(pdf_documents)
    return documents


def document_splitter(documents):
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=250,
        separators=["\n\n", "\n", " ",  ""]
    )
    documents = recursive_splitter.split_documents(documents)

    for i, doc in enumerate(documents):
        doc.metadata['source'] = doc.metadata['source'].split('/')[-1]
        doc.metadata['doc_id'] = i
    return documents


def make_vector_store(documents):
    embedding_model = GoogleGenerativeAIEmbeddings(
        model='models/text-embedding-004'
        )
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embedding_model
    )
    return vector_store


def make_chain_conversation():

    document = document_loader()
    document = document_splitter(document)
    vector_store = make_vector_store(document)

    chat = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
    )

    retriever = vector_store.as_retriever()

    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key='chat_history',
        output_key='answer'
        )
    chat_chain = ConversationalRetrievalChain.from_llm(
        llm=chat,
        memory=memory,
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )
    st.session_state['chain'] = chat_chain

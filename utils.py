import streamlit as st
from pathlib import Path
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

PDF_FOLDER = Path(__file__).parent / 'PDF'
MODEL_NAME = 'gemini-1.5-flash'
RETRIEVAL_SEARCH_TYPE = 'mmr'
RETRIEVAL_PARAMS = {'k': 5, 'fetch_k': 20}
PROMPT = '''
Você é um ótimo professor de programação, já atuou durante um tempo como full
stack no mercado e sabe tudo sobre códigos

Você recebeu diversos documentos sobre programação e deve auxililar o aluno
com suas duvidas, interprete os documentos e forneça as respostas com base 
neles

Utilize o contexto para responder as perguntas do usuário. 

Se você não souber a resposta simplesmente diga que não consegue responde-la,
sempre dê suas respostas em portugues do Brasil independente se houver outras
linguas no documento fornecido, sempre que houverem termos técnicos preserve 
a lingua original presente no documento

Contexto:
{context}

Conversa atual:
{chat_history}

Human:
{question}

Ai:
'''


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
        search_kwargs=RETRIEVAL_PARAMS
    )

    retriever = vector_store.as_retriever(
        search_type=RETRIEVAL_SEARCH_TYPE,
        search_kwargs=RETRIEVAL_PARAMS
    )

    prompt_template = PromptTemplate.from_template(
        template=PROMPT
    )

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
        verbose=True,
        combine_docs_chain_kwargs={'prompt': prompt_template}
    )
    st.session_state['chain'] = chat_chain

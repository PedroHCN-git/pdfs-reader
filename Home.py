import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from pathlib import Path
from time import sleep


PDF_FOLDER = Path(__file__).parent / 'PDF'
MEMORY = ConversationBufferMemory(return_messages=True)


def creat_chain_conversation():
    st.session_state['chain'] = True

    memory = MEMORY
    memory.chat_memory.add_ai_message('Oi')
    memory.chat_memory.add_ai_message('Eu sou uma LLM')

    st.session_state['memory'] = MEMORY
    sleep(2)


def sidebar():
    uploaded_pdfs = st.file_uploader(
        'Adicione seus arquivos PDF',
        type=['.pdf'],
        accept_multiple_files=True
        )
    if not uploaded_pdfs is None:
        for file in PDF_FOLDER.glob('*.pdf'):
            file.unlink()
        for pdf in uploaded_pdfs:
            with open(PDF_FOLDER / pdf.name, 'wb') as f:
                f.write(pdf.read())
    button_label = 'Inicializar ChatBot'
    if 'chain' in st.session_state:
        button_label = 'Atualizar ChatBot'
    if st.button(button_label, use_container_width=True):
        if len(list(PDF_FOLDER.glob('*.pdf'))) == 0:
            st.error('Por favor faça o upload de arquivos para inicializar')
        else:
            st.success('inicializando Chat Bot...')
            creat_chain_conversation()
            st.rerun()


def chat_window():
    st.header('Bem vindo ao Chat com PDFs', divider=True)

    if not 'chain' in st.session_state:
        st.error('Inicialize o ChatBot para começar')
        st.stop()

    # chain = st.session_state['chain']
    # memory = chain.memory

    memory = st.session_state['memory']
    messages = memory.load_memory_variables({})['history']

    container = st.container()
    for message in messages:
        chat = container.chat_message(message.type)
        chat.markdown(message.content)

    user_input = st.chat_input('digite uma mensagem...')
    
    if user_input:
        chat = container.chat_message('human')
        chat.markdown(user_input)
        chat = container.chat_message('ai')
        chat.markdown('Gerando uma resposta...')
        sleep(2)
        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message('oi é a LLM de novo')

        st.rerun()



def main():
    with st.sidebar:
        sidebar()
    chat_window()


if __name__ == '__main__':
    main()

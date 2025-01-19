import streamlit as st
from utils import make_chain_conversation, PDF_FOLDER
from dotenv import load_dotenv, find_dotenv
from time import sleep

_ = load_dotenv(find_dotenv())


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
            make_chain_conversation()
            st.rerun()


def chat_window():
    st.header('Bem vindo ao Chat com PDFs', divider=True)

    if not 'chain' in st.session_state:
        st.error('Inicialize o ChatBot para começar')
        st.stop()

    chain = st.session_state['chain']
    memory = chain.memory

    messages = memory.load_memory_variables({})['chat_history']

    container = st.container()
    for message in messages:
        chat = container.chat_message(message.type)
        chat.markdown(message.content)

    user_input = st.chat_input('digite uma mensagem...')
     
    if user_input:
        chat = container.chat_message('human')
        chat.markdown(user_input)
        chat = container.chat_message('ai')
        chain.invoke({'question': user_input})
        sleep(2)

        st.rerun()


def main():
    with st.sidebar:
        sidebar()
    chat_window()


if __name__ == '__main__':
    main()

import streamlit as st
from tempfile import NamedTemporaryFile
from pathlib import Path
from time import sleep

PDF_FOLDER = Path(__file__).parent / 'PDF'
files = list(PDF_FOLDER.glob('*.pdf'))


def load_pdf(files_path):
    for path in files_path:
        with open(path, 'rb') as f:
            with NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
                temp.write(f.read())
                temp_name = temp.name
            document = temp_name
        return document


def creat_chain_conversation():
    sleep(2)
    st.session_state['chain'] = True


def sidebar():
    st.button('Inicializar Chat Bot', use_container_width=True)
    print(load_pdf(files))


def main():
    with st.sidebar:
        sidebar()


if __name__ == '__main__':
    main()

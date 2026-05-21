__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from tempfile import NamedTemporaryFile

import streamlit as st
import chromadb

from query_data import classify_img, get_most_similar_chunks, create_response
from query_graph_db import create_graphrag_response
from utils import DB_PATH, get_config


client_db = chromadb.PersistentClient(path=DB_PATH)

# set title
st.title('tour-guide-ai-assistant-app')

# set header
st.header("Please upload an image")

# upload file
file = st.file_uploader("", type=["jpeg", "jpg", "png"])

if file:
    # display image
    st.image(file, use_column_width=True)

    with NamedTemporaryFile(dir='.') as f:
        f.write(file.getbuffer())
        image_path = f.name

        clf = classify_img(client_db, image_path)
        clf_ = clf.replace('_', ' ').title()

        config = get_config(clf)

        st.write(f'You are currently looking at the **{clf_}** !\n Is there anything you would like to know about it?')

        user_question = st.text_input(f'Ask a question about **{clf_}**:')

        # write agent response
        if user_question and user_question != "":
            with st.spinner(text="In progress..."):

                chunks, metadata = get_most_similar_chunks(client_db, user_question, clf)
                traditional_rag_response, sources = create_response(chunks, metadata, user_question)
                graph_rag_response, context_data = create_graphrag_response(config, query=user_question, mode="global")

                st.write(f"Traditional RAG:\n{traditional_rag_response}")
                st.write(f"Graph RAG response (global):\n{graph_rag_response}")

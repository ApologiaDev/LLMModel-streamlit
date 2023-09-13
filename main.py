
import json
import os
from zipfile import ZipFile
import tempfile

import streamlit as st
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

from util.modelhelpers import get_llm_model, get_embeddings_model


# load environment variables from .env
st.text('Enter necessary key: ')
open_ai_key = st.text_input('OpenAI Key', type='password')
huggingface_api_key = st.text_input('Huggingface API Key', type='password')
os.environ['OPENAI_API_KEY'] = open_ai_key
os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingface_api_key


# page
uploaded_model_file = st.file_uploader('Choose a model file (.zip)')
if uploaded_model_file is not None:
    # initialize context
    modeltmpdir = tempfile.mkdtemp()
    with ZipFile(uploaded_model_file, 'r') as zip_ref:
        zip_ref.extractall(modeltmpdir)
    config = json.load(open(os.path.join(modeltmpdir, 'config.json'), 'r'))
    llm = get_llm_model(config)
    embedding = get_embeddings_model(config)
    db = FAISS.load_local(modeltmpdir, embedding)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

# input
input_question_text = st.text_area('Question')

# get results
if st.button('Ask!'):
    if uploaded_model_file is None:
        st.write('Please upload a model!')
    else:
        response = qa({'query': input_question_text})
        st.write(response['result'])

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
# import torch
import base64
import textwrap
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
import os
from constant import CHROMA_SETTINGS

checkpoint = "LaMini-Flan-T5-783M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, device_map="auto", offload_folder="offload")


@st.cache_resource
def llm_pipeline():
    pipe = pipeline(task='text2text-generation', model=model,
                    tokenizer=tokenizer,
                    max_length=256,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.95)
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm


@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    db = Chroma(persist_directory="db", embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa


def process_answer(question):
    response = ''
    instruction = question
    qa = qa_llm()
    generated_response = qa(instruction)
    result = generated_response['result']
    return result, generated_response['source']


def main():
    st.title("LLM Chat App 🧑📄")
    with st.expander("About the app"):
        st.markdown(
            '''
            ## About
            This app is an LLM-popwered chatbot built using:
            - [Streamlit](https://streamlit.io/)
            - [LangChain](https://python.langchain.com)
            - [OpneAI](https://platform.openai.com/docs/models) LLM model
            '''
        )
        add_vertical_space(5)
    question = st.text_input("Ask questions about Bangladesh constitution")
    if question:
        result, source = process_answer(question)
        st.write(result)
        st.write(source)


if __name__ == '__main__':
    main()

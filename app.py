import os
import pickle
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
# from langchain.embeddings.ollama import OllamaEmbeddings
# sideber contents
with st.sidebar:
    st.title("LLM Chat App")
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


def main():
    st.header("Chat with any PDF")
    load_dotenv()
    pdf = st.file_uploader('Upload you PDF', type='pdf')
    # st.write(pdf.name)
    # Read PDF
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        # st.write(pdf_reader)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Embeddings

        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            # embeddings = OllamaEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        ## Take query
        query = st.text_input("Ask questions about Bangladesh constitution")

        if query:
            docs = VectorStore.similarity_search(query=query,k=3)

            llm = OpenAI(temperature=0)
            chain = load_qa_chain(llm=llm,chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs,question=query)
                print(cb)
            st.write(response)

        st.write(chunks)


if __name__ == '__main__':
    main()

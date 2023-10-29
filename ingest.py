from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import os
from constant import CHROMA_SETTINGS

persist_directory = 'db'


def main():
    for root, dirs, files in os.walk("pdfs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))
                # loader = PyPDFLoader(os.path.join(root,file))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    # Embeddings
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2")
    # Create vector Store
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    db.persist()
    db = None


if __name__ == '__main__':
    main()

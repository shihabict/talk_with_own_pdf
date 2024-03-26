from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from dotenv import dotenv_values
import os

secrets = dotenv_values()

os.environ['OPENAI_API_KEY'] = secrets['OPENAI_API_KEY']


class SimpleRAGApplication:

    def __init__(self, pdf_dir):
        self.documents = SimpleDirectoryReader(pdf_dir).load_data()
        self.index = VectorStoreIndex.from_documents(self.documents,show_progress=True)


    def read_pdfs_from_directory(self, pdf_dir):
        documents = SimpleDirectoryReader(pdf_dir).load_data()
        return documents

    def create_vector_index(self, documents):
        index = VectorStoreIndex.from_documents(documents)
        return index

    def query_vector_index(self, index, query):
        response = index.query(query)
        return response

    def main(self, query):
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query)

        return response


if __name__ == '__main__':
    app = SimpleRAGApplication('data')
    query = 'What is transformer?'
    response = app.main(query)
    print(response.response)

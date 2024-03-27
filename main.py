from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from dotenv import dotenv_values
import os

secrets = dotenv_values()

os.environ['OPENAI_API_KEY'] = secrets['OPENAI_API_KEY']

import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)


class SimpleRAGApplication:

    # def __init__(self, pdf_dir):
    #     self.documents = SimpleDirectoryReader(pdf_dir).load_data()
    #     self.index = VectorStoreIndex.from_documents(self.documents,show_progress=True)

    def store_in_memory(self, PERSIST_DIR):
        if not os.path.exists(PERSIST_DIR):
            # load the documents and create the index
            documents = SimpleDirectoryReader("data").load_data()
            self.index = VectorStoreIndex.from_documents(documents)
            # store it for later
            self.index.storage_context.persist(persist_dir=PERSIST_DIR)
        else:
            # load the existing index
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            self.index = load_index_from_storage(storage_context)

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
    PERSIST_DIR = 'storage'
    app = SimpleRAGApplication()
    query = 'What is transformer?'
    app.store_in_memory(PERSIST_DIR)
    response = app.main(query)
    print(response.response)

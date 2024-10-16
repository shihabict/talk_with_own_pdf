import os.path
from langchain.chains.llm import LLMChain
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_community.document_loaders.pdf import PyPDFLoader


class DocSummarizer:

    def __init__(self,ollama_model):
        self.prompt_template =  """
                                Write a long summary of the following document. 
                                Only include information that is part of the document. 
                                Do not include your own opinion or analysis.
                                
                                Document:
                                "{document}"
                                Summary:
                              """

        self.prompt_extended_post = """
                                    You are an expert who can write larger context document from small document.
                                    While writing document focus on the provided information and write extensively.
                                    Do not hallucinate much,
                                    maintain coherence and context of the writing.
        
                                    """
        # Define LLM Chain

        self.llm = ChatOllama(
            model=ollama_model,
            temperature=0,
            # other params...
        )

        self.prompt = PromptTemplate.from_template(self.prompt_template)
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)

        #Invoking the stuff documents chain
        # Create full chain

        self.stuff_chain = StuffDocumentsChain(
            llm_chain=self.llm_chain, document_variable_name="document"
        )


    def load_pdf(self,pdf_path):
        if os.path.exists(pdf_path):
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            return docs
        else:
            print('PDF not exist')

    def get_summary(self,pdf_path):
        docs = self.load_pdf(pdf_path)
        return self.stuff_chain.invoke(docs)

    def write_extended_document(self,pdf_path):
        docs = self.load_pdf(pdf_path)
        # return self.llm_chain.invoke()


if __name__ == '__main__':
    ollama_model = 'llama3.2:1b'
    pdf_path = 'dataset/AhmedSyed petition body.pdf'
    doc_summarizer = DocSummarizer(ollama_model=ollama_model)
    # doc_summarizer.load_pdf(pdf_path=pdf_path)
    summary = doc_summarizer.get_summary(pdf_path)
    print(summary['output_text'])
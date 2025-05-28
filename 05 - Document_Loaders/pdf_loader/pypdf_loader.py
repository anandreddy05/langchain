from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

model = ChatOllama(model="mistral:latest")
parser = StrOutputParser()

loader = PyPDFLoader('./SRS_COMPLETE.pdf')

docs = loader.load()

full_text = "\n".join([doc.page_content for doc in docs])


prompt = PromptTemplate(
    template="Generate the title and summary of the {text}",
    input_variables=['text']
)

chain = prompt | model | parser

response = chain.invoke({"text":full_text})
print(response)

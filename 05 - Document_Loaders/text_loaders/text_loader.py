from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

model = ChatOllama(model="mistral:latest")


loader = TextLoader('text.txt',encoding='utf-8')

docs = loader.load()

prompt = PromptTemplate(
    template="Generate an amazing Title and Summary of the following {text}",
    input_variables=['text']
)
parser = StrOutputParser()

chain = prompt | model | parser

response = chain.invoke({"text":docs[0].page_content})

print(response)
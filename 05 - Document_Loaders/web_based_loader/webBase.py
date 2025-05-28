from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
model = ChatOllama(model="mistral:latest")
parser = StrOutputParser()

url = "https://en.wikipedia.org/wiki/Universe"
loader = WebBaseLoader(url)
docs = loader.load()
print(len(docs))
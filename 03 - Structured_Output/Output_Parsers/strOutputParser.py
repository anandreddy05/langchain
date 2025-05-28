from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model="llama2:latest")

template1 = PromptTemplate(
    template="You are a helpful assistant create a detailed repont on {topic}",
    input_variables=["topic"]
)
template2 = PromptTemplate(
    template="Create a detailed summary in 5 lines on the report {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic":"Black Holes"})

print(result)
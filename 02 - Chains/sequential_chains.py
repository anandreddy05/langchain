from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time

model = ChatOllama(model="mistral:latest")
parser = StrOutputParser()


# prompt = PromptTemplate(
#     template="Generate 5 interesting facts about {topic} in new line with indexes",
#     input_variables=["topic"]
# )


# chain = prompt | model | parser

# response = chain.invoke({"topic":"Earth"})

# # print(response)

# chain.get_graph().print_ascii()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='Generate a 5 line detailed summary which should cover most of the given {report}',
    input_variables=['report']
)
start_time = time.time()

chain = prompt1 | model | parser | prompt2 | model | parser
response = chain.invoke({"topic":"Life on Mars"})

elapsed_time = time.time() - start_time

print(f"Response Time: {elapsed_time}")
print(response)
# Conditional Use Cases

from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableBranch, RunnableSequence,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

prompt1 = PromptTemplate(
    template = "Write a detailed on \n {topic}",
    input_variables=['text']
)
prompt2 = PromptTemplate(
    template = "summarize the following text \n {text}",
    input_variables=['text']
)

model = ChatOllama(model="mistral:latest")

parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1,model,parser)

branch_chain = RunnableBranch(
    (lambda x : len(x.split())>200,RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)
final_chain = RunnableSequence(report_gen_chain,branch_chain)

res = final_chain.invoke({"topic":"Gravity"})
print(res)
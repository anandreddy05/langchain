# Runnable Sequence -> sequential chain
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    temperature=0.7
)

prompt1 = PromptTemplate(
    template= "Generate a joke about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate a explanation about {text} in detail",
    input_variables=['topic']
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2,model,parser)
})

chain = joke_gen_chain | parallel_chain

res = chain.invoke({'topic':"Life" })

print(res)
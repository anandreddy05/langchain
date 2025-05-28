# Runnable Sequence -> sequential chain
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    temperature=0.7
)

prompt1 = PromptTemplate(
    template= "Generate a tweet about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate a LinkedIn post about {topic} in detail",
    input_variables=['topic']
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1,model,parser),
    'linkedin': RunnableSequence(prompt2,model,parser)
})

res = chain.invoke({'topic':"AI"})
print("Tweet: ",res['tweet'])
print("="*30)
print("LinkedIn",res['linkedin'])
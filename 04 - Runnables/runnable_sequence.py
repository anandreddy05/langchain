# Runnable Sequence -> sequential chain
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    temperature=0.7
)

prompt1 = PromptTemplate(
    template= "Generate a ine line joke on the {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Explain the given {text} in detail",
    input_variables=['text']
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()
chain = RunnableSequence(prompt1,model,parser,prompt2,model,parser)
res = chain.invoke({'topic':"AI"})
print(res)
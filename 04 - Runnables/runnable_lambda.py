from langchain.schema.runnable import RunnableLambda,RunnableParallel,RunnablePassthrough,RunnableSequence
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    temperature=0.7
)

prompt = PromptTemplate(
    template= "Generate a joke about {topic}",
    input_variables=['topic']
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt,model,parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count': RunnableLambda(lambda x: len(x.split()))
})

final_chain = RunnableSequence(joke_gen_chain,parallel_chain)

res = final_chain.invoke({"topic":"Jupiter"})

print(res)
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal


llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2", 
    task="text-generation",
    temperature=0.4
)

model = ChatHuggingFace(llm=llm)

class FeedBack(BaseModel):
    sentiment: Literal['positive','negative'] = Field(description="Give the sentiment of the feedback")
    
str_parser = StrOutputParser()
pydantic_parser = PydanticOutputParser(pydantic_object=FeedBack)
format_instruction = pydantic_parser.get_format_instructions()

prompt1 = PromptTemplate(
    template="""Classify the following feedback sentiment as either 'positive' or 'negative'.
Return your answer as a valid JSON with a single field called 'sentiment'.

Feedback: {feedback}

{format_instructions}

Return ONLY the JSON object with no additional text.""",
    input_variables=['feedback'],
    partial_variables={"format_instructions": format_instruction}
)

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback \n {feedback}",
    input_variables=["feedback"]
)
prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback \n {feedback}",
    input_variables=["feedback"]
)


classifier_chain = prompt1 | model | pydantic_parser

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | str_parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model | str_parser),
    RunnableLambda(lambda x: "Could not determine sentiment")
)


chain = classifier_chain | branch_chain

result = chain.invoke({"feedback":"This is a amazing phone"})
print(result)
from pydantic import BaseModel,Field
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
import huggingface_hub
import os
from dotenv import load_dotenv
from typing import Literal


load_dotenv()

hf_token = os.getenv("HUGGINGFACE_API")

if not hf_token:
    raise ValueError("Hugging Face API key is missing. Set HUGGINGFACE_API in .env")
huggingface_hub.login(hf_token)

class User(BaseModel):
    name:str = Field(description="Name of the person")
    age: int = Field(gt=18,description="Age of the person")
    city:str = Field(description="Name of the city the person belongs to")
    gender:Literal["male","female"] = Field(description="Gender of the person")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2", 
    task="text-generation",
    temperature=0.2  
)

model = ChatHuggingFace(llm=llm)
parser = PydanticOutputParser(pydantic_object=User)
format_instructions = parser.get_format_instructions()

template = PromptTemplate(
    template=(
        "Generate a fictional {gender} person from {place}. "
        "Provide the details in **valid JSON format** as follows:\n"
        "```\n{format_instructions}\n```\n"
        "Ensure the JSON contains real values, not a schema."
    ),
    input_variables=['place', 'gender'],
    partial_variables={'format_instructions': format_instructions}
)

# prompt = template.format(place="India",gender="male")
# result = model.invoke(prompt)
# print("Result: ",result)
# print("="*40)
# final_result = parser.parse(result.content)
# print("Final Result: ",final_result)
chain = template | model | parser

result = chain.invoke({'place':"Andhra Pradesh",'gender':'male'})

print(result)
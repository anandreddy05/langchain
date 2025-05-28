# No particular Schema in this
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import huggingface_hub
import os
from dotenv import load_dotenv
load_dotenv()

hf_token = os.getenv("HUGGINGFACE_API")
if not hf_token:
    raise ValueError("Hugging Face API key is missing. Set HUGGINGFACE_API in .env")
huggingface_hub.login(hf_token)

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    temperature=2
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name,age,and the city of a fictional preson \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# prompt = template.format()

# result = model.invoke(prompt)

# formattes_result = parser.parse(result.content)
# print(formattes_result)

chain = template | model | parser

result = chain.invoke({})
print(result)
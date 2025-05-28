# particular Schema in this
# no data validation
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser,ResponseSchema
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
    temperature=0.2  
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="fact_1",description="Fact 1 about the topic"),
    ResponseSchema(name="fact_2",description="Fact 2 about the topic"),
    ResponseSchema(name="fact_3",description="Fact 3 about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)
format_instructions = parser.get_format_instructions()
template = PromptTemplate(
    template="Give 3 fact about {topic} \n {format_instructions}",
    input_variables=['topic'],
    partial_variables={'format_instructions':format_instructions}
)

# prompt = template.format(topic='White Hole')

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

# print(final_result)
chain = template | model | parser

result = chain.invoke({'topic':'white hole'})
print(result)
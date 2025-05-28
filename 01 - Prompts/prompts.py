from langchain_ollama import ChatOllama
import streamlit as st
import time
from langchain_core.prompts import PromptTemplate

llm = st.selectbox("Select your desired model",["mistral:latest","deepseek-r1:7b","llama2:latest"])

model = ChatOllama(model=llm,
                   temperature=1)

st.title("AI Research Paper Summarizer") 
st.write("Enter a prompt and click the button to get a response.")

st.write("Select Your Desired Model")

template = PromptTemplate(
    input_variables=['user_input'],
    template="""
     Summarize the uploaded research paper Name by the user
    
    Instructions:
    1. If the titles research contains **mathematical operations**, explain them in detail.
    2. Include relevant **mathematical equations** if they appear in the research paper or title.
    3. Use **relatable analogies** to simplify complex ideas.
    4. If the research paper provides specific information, include it. Otherwise, state: **"Insufficient information available."**
    5. **DO NOT hallucinate.** Ensure the summary is accurate, simple, and easy to understand.
    """
    )


user_input = st.text_input("Enter Topic Title")

if st.button("Summarize"):
    with st.spinner(text=f"{llm} in Progress...",show_time=False):
        if user_input:
            prompt = template.invoke({'user_input':user_input})

            start_time = time.time()
            response = model.invoke(prompt)
            elapsed_time = time.time() -  start_time
            st.success(f"Completed in {elapsed_time:.2f} seconds")
            st.write(response.content)
        else:
            st.error("Please Enter Topic before summarizing...")


from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_ollama import ChatOllama
import streamlit as st
import time

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "mistral:latest"

if "model" not in st.session_state:
    st.session_state.model = ChatOllama(model=st.session_state.selected_model)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
         SystemMessage(content="You are a helpful assistant make the conversation as simple as possible")
    ]
    

selected_model = st.selectbox("Select your desired model",["mistral:latest","llama2:latest","deepseek-r1:7b"])

if selected_model != st.session_state.selected_model:
    st.session_state.selected_model = selected_model
    st.session_state.model = ChatOllama(model=selected_model)


st.title("Simple ChatBot with Chat History")

user_input = st.text_input("User: ")

if st.button("Send"):
    with st.spinner(text=f"{st.session_state.selected_model} is processing",show_time=False):
        if user_input:
            start_time = time.time()
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            response = st.session_state.model.invoke(st.session_state.chat_history)
            st.session_state.chat_history.append(AIMessage(content=response.content))

            elapsed_time = time.time() - start_time
            st.success(f"Completed in {elapsed_time:.2f} seconds")
            st.write(f"User: {user_input}")
            st.write(f"AI: {response.content}")
        else:
            st.error('Please enter the prompt')

st.write("=======================================================================================")
st.write("### Chat History")
if len(st.session_state.chat_history) > 1:
    for i, message in enumerate(st.session_state.chat_history[1:], start=1):  
        role = "User" if isinstance(message, HumanMessage) else "AI"
        st.write(f"{i}. **{role}:** {message.content}")

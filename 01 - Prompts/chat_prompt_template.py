# Use ChatPromptTemplate when we have list of messages

from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history')
    ('human', '{query}')
    ])

with open('chat_history.txt') as f:
    f.readlines()
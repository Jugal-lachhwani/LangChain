from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage

template = ChatPromptTemplate([
    ('system','You are a heplful customer support agent'),
    MessagesPlaceholder(variable_name = 'chat_history'),
    ('human','{querry}')
])

Chat_history = []

with open('Prompts\chat_histoty.txt') as f:
    Chat_history.extend(f.readlines())
    
prompt = template.invoke({'chat_history':Chat_history,'querry':'Where is my order'})

print(prompt)
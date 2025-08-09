from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}')
])

prompt = template.invoke({'domain':'cricket','topic':'Dusra'})

print(prompt)
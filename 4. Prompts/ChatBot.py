from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest',
                               model_kwargs={"max_output_tokens": 20})

chat_history = [
    SystemMessage(content = 'You are very funny assistant')
]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content = user_input))
    if user_input == 'exit':
        print('Ok, See you later')
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content = result.content))
    print('AI: ',result.content)
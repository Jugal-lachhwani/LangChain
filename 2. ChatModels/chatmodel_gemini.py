from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest')

result = model.invoke('what is the capital of Delhi')

print(result.content)
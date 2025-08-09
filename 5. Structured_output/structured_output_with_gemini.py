from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

class Intro(BaseModel):
    name: str
    age: int

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

structured_model = model.with_structured_output(Intro)

text = "My name is Jugal and I am 18"

result = structured_model.invoke(text)

print(result)

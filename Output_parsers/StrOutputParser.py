from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

template1 = PromptTemplate(
    template = 'Write a long story about {topic}',
    input_variables = ['topic']
)

template2 = PromptTemplate(
    template = 'give the summary of {story}',
    input_variables = ['story']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

text = 'Harry Potter and the Philosopher\'s Stone'

chain_result = chain.invoke({'topic': text})

print(chain_result)
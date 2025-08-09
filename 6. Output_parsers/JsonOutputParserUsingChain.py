from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

parser = JsonOutputParser()

template1 = PromptTemplate(
    template = 'Write a long story about {topic} \n {format_instructions}',
    input_variables = ['topic'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

template2 = PromptTemplate(
    template = 'give the summary of {story} \n {format_instructions}',
    input_variables = ['story'],
    partial_variables={'format_instructions': parser.get_format_instructions() + " Give only first 3 chapters."}
)

chain = template1 | model | parser | template2 | model | parser

text = 'Harry Potter and the Philosopher\'s Stone'

chain_result = chain.invoke({'topic': text})

print(chain_result)
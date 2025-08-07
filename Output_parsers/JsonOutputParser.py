from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

parser = JsonOutputParser()

template1 = PromptTemplate(
    template = 'Give the name and age of fictional character \n {format_instructions}',
    input_variables = [],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

chain = template1 | model | parser

chain_result = chain.invoke({})

print(chain_result)
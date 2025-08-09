from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

url = "https://en.wikipedia.org/wiki/Mukesh_Ambani"

loader = WebBaseLoader(url)

docs = loader.load()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

template = PromptTemplate(
    template = 'Answer the following quetion {question} based on the following context: {context} if the ans with your own training knowlege',
    input_variables = ['question', 'context']
)

parser = StrOutputParser()

chain = template | model | parser

final_result = chain.invoke({'question': 'Which is the most earning film in India?', 'context': docs[0].page_content})

print(final_result)

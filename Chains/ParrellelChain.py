from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel,RunnableSequence,RunnablePassthrough

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

template1 = PromptTemplate(
    template = 'Tell a joke on {topic}',
    input_variables = ['topic']
)

template2 = PromptTemplate(
    template = 'Give me a brief explanation of {topic}',
    input_variables = ['topic']
)

parser = StrOutputParser()

chainExplain = template1 | model | parser | template2 | model | parser

chain = RunnableParallel({
    ('Joke'): template1 | model | parser,
    ('Explanation'):template2 | model | parser
})

final_result = chain.invoke({'topic': 'python'})

print(final_result)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableBranch,RunnablePassthrough,RunnableLambda
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

class SentimentAnalysis(BaseModel):
    sentiment: Literal['positive', 'negative', 'neutral'] = Field(description='Sentiment of the text')
    text: str = Field(description='Text of the feedback')

pydparser = PydanticOutputParser(pydantic_object=SentimentAnalysis)

template0 = PromptTemplate(
    template= 'Give the sentiment of the feedback and pass the text also : {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': pydparser.get_format_instructions()}
)

chain = template0 | model | pydparser

template1 = PromptTemplate(
    template = 'Write an appropriate response to this positive feedback dont give the respose choose it by yourself : {feedback}',
    input_variables = ['feedback']
)

template2 = PromptTemplate(
    template = 'Give the reply a neutral sentiment of {feedback}',
    input_variables = ['feedback']
)
template3 = PromptTemplate(
    template = 'Give the reply a negative sentiment of {feedback}',
    input_variables = ['feedback']
)

parser = StrOutputParser()

Conditional_branch = RunnableBranch(
    (lambda x:x.sentiment == 'positive', RunnableSequence(template1, model, parser)),
    (lambda x:x.sentiment == 'negative', RunnableSequence(template2, model, parser)),
    (lambda x:x.sentiment == 'neutral', RunnableSequence(template3, model, parser)),
    RunnableLambda(lambda x: x)
)

main_chain = chain | Conditional_branch

final_result = main_chain.invoke({'feedback': 'This product is amazing!'})

print(final_result)
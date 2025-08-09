from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import CSVLoader
from dotenv import load_dotenv

load_dotenv()

loader = CSVLoader(file_path ="E:\Data_Science\Dataset_notebooks\Dataset\Titanic-Dataset.csv")

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

docs = loader.load()

template = PromptTemplate(
    template = 'Answer the following question {question} based on the following context: {context}',
    input_variables = ['question', 'context']
)

# print(docs)
# print(len(docs))

i = 0

content = []

for doc in docs:
    i+=1
    # print(doc.page_content)
    content.append(doc.page_content)
    
print(len(content))

content = ''.join(content)
print(len(content))

# content = ''.join([doc.page_content for doc in docs])

# print(content)
# print(len(content))
parser = StrOutputParser()

chain = template | model | parser

final_result = chain.invoke({'question': 'How many people are traveling with 3 siblings or spouse  in the dataset?', 'context': content})

print(final_result)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader('Docs/text.txt', encoding='utf-8')

docs = loader.load()

spitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 10, 
)

result = spitter.split_documents(docs)

print(result)


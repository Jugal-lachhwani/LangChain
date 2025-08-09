from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader('Docs/text.txt', encoding='utf-8')

docs = loader.load()

spitter = CharacterTextSplitter(
    chunk_size = 10,
    chunk_overlap = 0,
    separator = '\n',   
)

result = spitter.split_documents(docs)

print(result)


from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

vectorstore = Chroma.from_documents(documents, embeddings, collection_name="my_collection")

retriver = vectorstore.as_retriever(search_kwargs={"k": 2})

querry = 'What is Chroma used for?'

results = retriver.invoke(querry)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
    
print("-----MMR------")

retriever_mmr = vectorstore.as_retriever(
    search_type="mmr",                   # <-- This enables MMR
    search_kwargs={"k": 3, "lambda_mult": 0.1}  # k = top results, lambda_mult = relevance-diversity balance
)

results_mmr = retriever_mmr.invoke(querry)

for i, doc in enumerate(results_mmr):
    print(f"\n--- MMR Result {i+1} ---")
    print(doc.page_content)

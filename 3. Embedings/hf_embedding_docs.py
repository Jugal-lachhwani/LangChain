from langchain_huggingface import HuggingFaceEmbeddings

embedings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

Doc = ['Delhi is the capital of india',
       'Ahmedabad is the best city of india',
       'Kolkatta is the biggest city of india']

vector = embedings.embed_documents(Doc)

print(str(vector))

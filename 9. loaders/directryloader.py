from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

loader = DirectoryLoader(
    path = "E:\\books\\pdfs",
    glob = '**/*.pdf',
    loader_cls = PyPDFLoader
)

docs = loader.load()

# docs = loader.lazy_load()

for doc in docs[:10]:
    print(doc.metadata)
    print(doc.page_content.encode('utf-8', 'replace').decode('utf-8'))
    print("========================================")
    
print(len(docs))
    
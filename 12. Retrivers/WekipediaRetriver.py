from langchain_community.retrievers import WikipediaRetriever


retriver = WikipediaRetriever()

querry = 'Who is the richest person in the world'

docs = retriver.invoke(querry)

print(docs)
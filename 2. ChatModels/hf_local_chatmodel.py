from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
from transformers import pipeline

os.environ['HF_Home'] = "E:"

llm = HuggingFacePipeline(pipeline = 
    pipeline(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    model_kwargs={"temperature": 0.5}
)
)

model = ChatHuggingFace(llm=llm)

result = model.invoke('Anything but not nothing')

print(result.content)
## Design and Implementation of LangChain Expression Language (LCEL) Expressions

### AIM:
To design and implement a LangChain Expression Language (LCEL) expression that utilizes at least two prompt parameters and three key components (prompt, model, and output parser), and to evaluate its functionality by analyzing relevant examples of its application in real-world scenarios.

### PROBLEM STATEMENT:

### DESIGN STEPS:

#### STEP 1:

#### STEP 2:

#### STEP 3:

### PROGRAM:
SIMPLE CHAIN
```
import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # Read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "Tell me about {topic}."
)

model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser

# Change the topic here
response = chain.invoke({"topic": "The Taj Mahal"})
print(response)
```
COMPLEX CHAIN
```
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

model = ChatOpenAI()
output_parser = StrOutputParser()

# Create vector database
vectorstore = DocArrayInMemorySearch.from_texts(
    [
        "The Taj Mahal is one of the most famous monuments in India and symbolizes love.",

        "Trees help the environment by producing oxygen, reducing pollution, and providing shade.",

        "Drinking enough water every day keeps the body hydrated and supports overall health.",

        "Computers are electronic devices used for learning, communication, and solving problems."
    ],
    embedding=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever()

# Test retrieval
print(retriever.get_relevant_documents("What is Artificial Intelligence?"))
print(retriever.get_relevant_documents("Why are trees important?"))

template = """
Answer the question based only on the following context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)



chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | output_parser

# Question 1
result1 = chain.invoke({
    "question": "Why is the Taj Mahal famous?"
})

print(result1)

# Question 2
result2 = chain.invoke({
    "question": "How do trees help us?"
})

print(result2)

# Question 3
inputs = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
})

print(inputs.invoke({
    "question": "Why should we drink plenty of water?"
}))
```


### OUTPUT:
SIMPLE CHAIN

<img width="528" height="187" alt="image" src="https://github.com/user-attachments/assets/91abb06c-baf4-4150-b0e0-e0310de58044" />

COMPLEX CHAIN

<img width="577" height="535" alt="image" src="https://github.com/user-attachments/assets/a6638aab-2fc9-40a4-97dd-d7ae47285e53" />

### RESULT:

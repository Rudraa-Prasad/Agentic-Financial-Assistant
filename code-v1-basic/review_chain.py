import os

from langchain.chains import RetrievalQA
#from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_groq import ChatGroq
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

groq_api_key =  os.getenv("GROQ_API_KEY")
from neo4j_connection import neo4j_vector_index

# Prompt Template
review_template = """
You are an assistant that uses customer reviews to answer questions 
about their experiences at different bank branches. Use the following 
context to answer. Be precise and detailed. Do not guess or fabricate any answer.
If the answer is not in the context, say "I don't know".

{context}
"""


review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"], template=review_template
    )
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["question"], template="{question}")
)
messages = [review_system_prompt, review_human_prompt]

review_prompt = ChatPromptTemplate(
    input_variables=["context", "question"], messages=messages
)

reviews_vector_chain = RetrievalQA.from_chain_type(
    llm=ChatGroq(model_name="llama3-8b-8192", api_key=groq_api_key,temperature=0),
    chain_type="stuff",
    retriever=neo4j_vector_index.as_retriever(k=12),
)
reviews_vector_chain.combine_documents_chain.llm_chain.prompt = review_prompt

# retrieval_qa_chain --> combine_docs --> run_llm_chain --> with_custom_prompt

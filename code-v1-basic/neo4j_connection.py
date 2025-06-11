import os
from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

NEO4J_CONFIG = {
    "url": os.getenv("NEO4J_URI"),
    "username": os.getenv("NEO4J_USERNAME"),
    "password": os.getenv("NEO4J_PASSWORD"),
}

# Shared Neo4j graph connection
graph = Neo4jGraph(**NEO4J_CONFIG)

# Shared vector index connection
neo4j_vector_index = Neo4jVector.from_existing_graph(
    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    index_name="bank_reviews",
    node_label="Review",
    text_node_properties=["review"],
    embedding_node_property="embedding",
    **NEO4J_CONFIG  # Reuse config
)

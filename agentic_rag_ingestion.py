from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import os
load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"

]

#Loading URLs into LangChain documents
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

doc_splits = text_splitter.split_documents(docs_list)

embeddings = OpenAIEmbeddings(openai_api_type=os.environ["OPENAI_API_KEY"])

vectorstore = PineconeVectorStore.from_documents(doc_splits,embeddings,index_name=os.environ['INDEX_NAME'],namespace="agentic-rag")

retriever = PineconeVectorStore(embedding = embeddings,index_name=os.environ['INDEX_NAME'],namespace="agentic-rag").as_retriever()

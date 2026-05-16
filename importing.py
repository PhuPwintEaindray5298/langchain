import asyncio
import os
import ssl
from typing import Any, Dict, List
import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import Colors, log_error, log_header, log_info, log_success, log_warning

load_dotenv()

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10,
)

vectorstore = PineconeVectorStore(
    index_name="langchain-doc-index", embedding=embeddings
)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap()
tavily_crawl = TavilyCrawl()


async def main():
    """Main async function to orchestratethe entire process"""
    log_header("Documentation Ingestion Pipeline")
    log_info(
        "TavilyCrawl: Starting to Crawl documentation from https://python.langchain.com/",
        Colors.PURPLE,
    )
    # Crawling the documentation site
    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/",
            "max_depth": 3,
            "extract_depth": "advanced",
            "instructions": "content on ai agents",
        }
    )
    all_docs = [
        Document(page_content=result["raw_content"], metadata={"source": result["url"]})
        for result in res["results"]
    ]
    log_success(
        f"TavilyCrawl: Successfully crawled {len(all_docs)} URLs from documentation site"
    )

    # Split documents into chunks
    log_header("Document Chunking Phase")
    log_info(
        f"Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks. from {len(all_docs)} documents"
    )

    await index_documetns_async(splitted_docs, batch_size=300)

    log_header("Pipeline Complete")
    log_success("Documentation ingestion pipeline finished successfully")
    log_info(" Summary:", Colors.BOLD)
    log_info(f"Documents mapped: {len(all_docs)}")
    log_info(f"Chunks created: {len(splitted_docs)}")


async def index_documetns_async(documents: List[Document], batch_size: int = 50):
    """Process documetns in batches asynchronously"""
    log_header("Vector Storage Phase")
    log_info(
        f"Vectorstore Indexing: Preparing to add {len(documents)} documents in vector store",
        Colors.DARKCYAN,
    )

    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]
    log_info(
        f"VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.add_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    # Process batches concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


if __name__ == "__main__":
    asyncio.run(main())

import os
import logging
import time
from pathlib import Path
import uuid
from typing import List

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from schemas import Chunk
from config import settings

# Tenacity imports for structural API error throttling guards
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from langchain_google_genai._common import GoogleGenerativeAIError


logger = logging.getLogger("RAG-Pipeline")

# Rate limit safeguards handle API call per minitue
BATCH_SIZE = 20         # stay well under 100 requests/min limit
DELAY_BETWEEN_BATCHES = 15                        # second reset the between batch windows

# create Embedding Model object
_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-004",
    api_key=settings.google_api_key,
    output_dimensionality=768,
)

# Local vector databse store use chroma db
_vectorstore = Chroma(
    collection_name=settings.pg_collection_name,  # settings field as the collection name
    embedding_function=_embeddings,
    persist_directory="./chroma_db"   # local folder, created automatically
)

# retry decorater to handle request
@retry(
        retry=retry_if_exception_type(GoogleGenerativeAIError),
        wait=wait_exponential(multiplier=2,min=10,max=60),
        stopt=stop_after_attempt(5),
)

def _add_batch_with_retry(docs: List[Document],ids: List[str]) -> None:
     """
    Executes the actual document insertion pass into Chroma. 
    Guarded by exponential backoff to handle rate limits gracefully.
    """
     _vectorstore.add_documents(docs,ids=ids)

def add_chunk(chunks:list[Chunk]) -> None:
    """
    Transforms your structural Pydantic chunk models down into baseline core LangChain documents,
    computes embedding matrices via Gemini cloud TPUs, and commits them to your local disk space.
    """
    # raise warring if chunks are empty
    if not chunks:
        logger.warning("No Chunks provided for databases injection loop.")
        return

     # Map your chunks into baseline core LangChain Document structures
    docs = [
        Document(
            page_content=chunk.content,
            metadata={
                "chunk_id"   : chunk.id,
                "source_type": chunk.source.source_type,
                "source_id"  : chunk.source.source_id,
                "position"   : chunk.source.position,
                **chunk.source.extra,
            }
        )
        for chunk in chunks
    ]

    ids = [chunk.id for chunk in chunks]

    total_chunks = len(docs)

    logger.info(f"Generating embedding for {total_chunks} text fragment via embedding model...")
    
    for i in range(0,total_chunks,BATCH_SIZE):
        batch_docs = docs[i : i + BATCH_SIZE]
        batch_ids = ids[i : i + BATCH_SIZE]

        current_batch_num = (i // BATCH_SIZE) + 1
        total_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"Streaming batch {current_batch_num}/{total_batches} ({len(batch_docs)} chunks)...")

        # Trigger the network upload sequence under the tenacity backoff umbrell
        _add_batch_with_retry(batch_docs,batch_ids)

        # Only sleep we have more batches remaining to process
        if i + BATCH_SIZE < total_chunks:
            logger.info(f"Throttling request pipeline. Resting for {DELAY_BETWEEN_BATCHES} seconds...")
            time.sleep(DELAY_BETWEEN_BATCHES)


    logger.info("Vector rows successfully committed locally to './chroma_db'.")



# Search 
def search(query: str, top_k: int = 5) -> list[tuple[Document,float]]:
    """
    Queries your local Chroma data files using mathematical distance equations 
    to fetch the highest matching contextual text rows.
    """
    logger.info(f"Executing vector search for query match profiles: '{query}'")
    return _vectorstore.similarity_search_with_score(query,k=top_k)

from pydantic import BaseModel, Field
from typing import Literal
from schemas.document import Chunk

class RAGQuery(BaseModel):

    question: str = Field(...,min_length=1)
    top_k: int = 5

class RetrieverDocs(BaseModel):
    chunk: Chunk
    score: float
    retrieval_method: Literal["bm25","vector","hybrid"] = "vector"
from pydantic import BaseModel
from schemas.retrieval import RetrieverDocs

class RAGAnswer(BaseModel):
    answer: str
    sources: list[RetrieverDocs]
    confidence: float | None = None
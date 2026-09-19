from retrieval.vector_retriever import search
from generation.chains import chain
from schemas import RetrieverDocs, Chunk, SourceInfo

def answer_question(question: str, top_k: int = 5):
    results = search(question,top_k=top_k)

    context_parts = []
    retrieved_docs = []

    for doc,score in results:
        context_parts.append(f"[{doc.metadata["source_id"]}] {doc.page_content}")
        retrieved_docs.append(
            RetrieverDocs(
                chunk=Chunk(
                    id=doc.metadata["chunk_id"],
                    content=doc.page_content,
                    source=SourceInfo(
                        source_type=doc.metadata["source_type"],
                        source_id=doc.metadata["source_id"],
                        position=doc.metadata["position"]
                    ),
                ),
                score=score,
                retrieval_method="vector",

            )
        )

    context = "\n\n".join(context_parts)
    result = chain.invoke({"context":context,"question":question})
    result.sources = retrieved_docs
    return result


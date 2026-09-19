from langchain_core.prompts import ChatPromptTemplate
from generation.llm import get_llm
from schemas import RAGAnswer

# instruction for system
SYSTEM_PROMPT = (
    "Answer the question using ONLY the provided context. "
    "If the context doesn't contain the answer, say so — do not guess."
    "Cite the source_id for every claim you make." 
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",SYSTEM_PROMPT),
        ("human","Context:\n {context} \n\nQuestion:{question}"),
    ]
)

llm = get_llm().with_structured_output(RAGAnswer)

chain = prompt | llm
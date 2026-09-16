import logging
from pathlib import Path
import uuid

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    WebBaseLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from schemas import Chunk, SourceInfo

logger = logging.getLogger("RAG_pipeline")

EXTENSION_MAP = {
    ".txt" : ("text_file",TextLoader),
    ".pdf" : ("pdf",PyPDFLoader),
    ".docs": ("docs",Docx2txtLoader),
    ".md"  : ("markdown",UnstructuredMarkdownLoader)
}

def get_loader_for_file(file_path: str):

    # Get Extionsion of file
    ext = Path(file_path).suffix.lower()

    if ext not in EXTENSION_MAP:
        raise ValueError(f"no loader configured for extension: {ext} ")

    source_type, loader_cls = EXTENSION_MAP[ext]
    return source_type, loader_cls(file_path) 

def load_and_chunk_file(file_path: str) -> list[Chunk]:
    source_type, loader = get_loader_for_file(file_path)
    docs = loader.load()

    if not any(doc.page_content.strip() for doc in docs):
        logger.warning(f"'{file_path}' produced no extractable text (likely a scanned/image PDF). "
                        "OCR support isn't wired up yet — skipping.")

    return _split_into_chunks(docs,source_type,source_id=file_path)

def load_and_chunk_url(url: str) -> list[Chunk]:
    loader = WebBaseLoader(url)
    docs = loader.load()
    return _split_into_chunks(docs,"web_page",source_id=url)


def _split_into_chunks(docs: list[Document], source_type: str, source_id: str) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    chunks = []

    for i, doc in enumerate(split_docs):
        extra_info = {
                **{k:v for k,v in (doc.metadata.copy() if doc.metadata else {}).items() if k not in ["page","source"]}
            }
        chunks.append(
            Chunk(
                id=str(uuid.uuid4()),
                content=doc.page_content,
                source=SourceInfo(
                    source_type=source_type,
                    source_id=source_id,
                    position=doc.metadata.get("page",i),
                    extra= extra_info,
                )
            )
        )

    return chunks


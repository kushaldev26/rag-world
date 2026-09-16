# use pdf multi loader startegy
import os
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
import numpy as np

# Setup professional Logging
logger = logging.getLogger("rag_pipeline")

EXTENSION_MAP = {
  ".txt": ("text_file", TextLoader),
  ".docx": ("docx", Docx2txtLoader),
  ".md": ("markdown", UnstructuredMarkdownLoader),

}

# load pdf with handle fallback condition
def load_pdf_with_fallback(file_path: str) -> list[Document]:
  
  """
  Tries to read the PDF quickly using standard text extraction layers.
  If it detects the PDF is an image-only scan or has bad encodings 
  (empty content strings), it gracefully falls back to OCR/pypdfium2 routing.
  """ 

  logger.info(f"Attempting standard PDF load for: {file_path}")
  try:
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # check if the extracted text layer is completely empty across pages
    total_text_len = sum(len(doc.page_content.strip()) for doc in docs)

    if total_text_len > 0:
      logger.info("Successfully extract text via standard PDF loader.")
      return docs
    
    logger.warning("⚠️ Standard PDF loader returned empty pages. Switching to OCR fallback strategy...")

  except Exception as e:
    logger.error(f"Stanadard PDF loader crashed: {str(e)}. Attempting OCR fallback strategy...")

  # --- OCR Fallback Strategy (Activated only if simpler methods fail) ---
  try:
      import pypdfium2 as pdfium
      import easyocr

      pdf = pdfium.PdfDocument(file_path)
      reader = easyocr.Reader(['en'])
      ocr_docs = []

      for page_num in range(len(pdf)):
        page = pdf[page_num]
        bitmap = page.render(scale=2) # High resolusion scale for better OCR reading
        pil_img = bitmap.to_pil()

        img_np = np.array(pil_img)

        # Extract word directly from the page impage array
        result = reader.readtext(img_np, detail=0)
        page_text=" ".join(result)

        ocr_docs.append(
            Document(
                page_content=page_text,
                metadata={"page":page_num, "source":file_path}
            )
        )
      return ocr_docs
    
  except ImportError as e:
      logger.critical(f"OCR Fallback failed missing critical packages: {str(e)}")
      raise RuntimeError(
        "PDF text extraction failed completely. Your environment is missing OCR requirements."
        "Please run: uv add pypdfium2 easyocr"  
      ) from e
  except Exception as e:
    logger.critical(f"Unexpected error during OCR rendering: {str(e)}")
    raise RuntimeError(f"Failed to process scanned PDF during OCR pass: {str(e)}") from e


def get_loader_for_file(file_path: str):
  
  ext = Path(file_path).suffix.lower()

  if ext == '.pdf':
    # Hand off control to the multi-loader fallback manager
        return "pdf", None

  if ext not in EXTENSION_MAP:
    raise ValueError(f"No loader configured extension:{ext}")

  source_type, loader_cls = EXTENSION_MAP[ext]
  return source_type, loader_cls(file_path)

def load_and_chunk_file(file_path: str) -> list[Chunk]:
  source_type, loader = get_loader_for_file(file_path)

  if source_type == "pdf":
    docs = load_pdf_with_fallback(file_path)
  else:
    docs = loader.load()

  return _split_into_chunks(docs,source_type, source_id=file_path)


# spling into chunks
def _split_into_chunks(docs: list[Document],source_type: str, source_id: str) -> list[Chunk]:
  splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
  split_docs = splitter.split_documents(docs)

  chunks = []

  for i, doc in enumerate(split_docs):
    # 1. Extract raw metadata dict from LangChain document
    raw_metadata = doc.metadata.copy() if doc.metadata else {}
    # 2. Packge all extra fields into the expexted dictionary Layout
    # If 'page' exits, it stay inside the dictionary for down stream tracking
    extra_info = {
        "page_number": raw_metadata.get("page", i),
        **{k:v for k,v in (doc.metadata.copy() if doc.metadata else {}).items() if k not in ["page","source"]}
    }

    # 3. Securely Validation
    chunks.append(
      Chunk(
          id = str(uuid.uuid4()),
          content=doc.page_content,
          source=SourceInfo(
              source_type=source_type,
              source_id=source_id,
              extra=extra_info,
          ),
      )
    )
  return chunks
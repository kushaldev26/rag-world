from pydantic import BaseModel,Field
from datetime import datetime,timezone
from typing import Literal
from rich import print

class SourceInfo(BaseModel):
  """Where this chunk came from - generic across source types. """
  
  source_type: Literal["code_repo","pdf","web_page","markdown","text_file","docs","other"]
  source_id: str  # file path, URL, repo name — whatever identifies the origin
  title: str | None = None   # doc title / page title / filename
  ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


  # Optinal, source-specific
  extra: dict = Field(default_factory=dict)
  # e.g. for code: {"repo_name": "...", "commit_hash": "...", "language": "python"}
  # e.g. for web:  {"url": "...", "domain": "..."}
  # e.g. for pdf:  {"page_number": 4, "author": "..."}


class Chunk(BaseModel):
  id: str
  content: str
  chunk_index: int = 0
  source: SourceInfo


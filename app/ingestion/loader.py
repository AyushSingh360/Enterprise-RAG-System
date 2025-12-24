"""
Document loaders for various file types: PDF, DOCX, Markdown, and SQL.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Any
import structlog

from llama_index.core.schema import Document
from llama_index.core import SimpleDirectoryReader

from app.schemas.documents import DocumentType

logger = structlog.get_logger(__name__)


class BaseLoader(ABC):
    """Abstract base class for document loaders."""
    
    @abstractmethod
    async def load(self, source: str, metadata: dict[str, Any]) -> list[Document]:
        """Load documents from a source."""
        pass


class PDFLoader(BaseLoader):
    """Loader for PDF documents."""
    
    async def load(self, file_path: str, metadata: dict[str, Any]) -> list[Document]:
        """Load a PDF file and extract text with page info."""
        from pypdf import PdfReader
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        documents: list[Document] = []
        reader = PdfReader(str(path))
        
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text.strip():
                doc_metadata = {
                    **metadata,
                    "source": path.name,
                    "file_path": str(path),
                    "page_number": page_num,
                    "total_pages": len(reader.pages),
                    "document_type": DocumentType.PDF.value,
                }
                documents.append(Document(text=text, metadata=doc_metadata))
        
        logger.info("Loaded PDF", file=path.name, pages=len(documents))
        return documents


class DOCXLoader(BaseLoader):
    """Loader for DOCX documents."""
    
    async def load(self, file_path: str, metadata: dict[str, Any]) -> list[Document]:
        """Load a DOCX file and extract text."""
        from docx import Document as DocxDocument
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"DOCX file not found: {file_path}")
        
        doc = DocxDocument(str(path))
        paragraphs: list[str] = []
        current_section = ""
        
        for para in doc.paragraphs:
            if para.style.name.startswith("Heading"):
                current_section = para.text
            paragraphs.append(para.text)
        
        full_text = "\n".join(paragraphs)
        
        doc_metadata = {
            **metadata,
            "source": path.name,
            "file_path": str(path),
            "document_type": DocumentType.DOCX.value,
        }
        
        logger.info("Loaded DOCX", file=path.name, paragraphs=len(paragraphs))
        return [Document(text=full_text, metadata=doc_metadata)]


class MarkdownLoader(BaseLoader):
    """Loader for Markdown content."""
    
    async def load(self, content: str, metadata: dict[str, Any]) -> list[Document]:
        """Load markdown content and extract sections."""
        import markdown
        from bs4 import BeautifulSoup
        
        html = markdown.markdown(content)
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text()
        
        doc_metadata = {
            **metadata,
            "document_type": DocumentType.MARKDOWN.value,
        }
        
        logger.info("Loaded Markdown", length=len(text))
        return [Document(text=text, metadata=doc_metadata)]


class SQLLoader(BaseLoader):
    """Loader for SQL database content."""
    
    async def load(
        self,
        connection_string: str,
        query: str,
        metadata: dict[str, Any]
    ) -> list[Document]:
        """Execute SQL query and convert results to documents."""
        from sqlalchemy import create_engine, text
        
        engine = create_engine(connection_string)
        
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
            columns = result.keys()
        
        documents: list[Document] = []
        for i, row in enumerate(rows):
            row_dict = dict(zip(columns, row))
            text_content = "\n".join([f"{k}: {v}" for k, v in row_dict.items()])
            
            doc_metadata = {
                **metadata,
                "source": f"sql_row_{i+1}",
                "row_index": i,
                "document_type": DocumentType.SQL.value,
            }
            documents.append(Document(text=text_content, metadata=doc_metadata))
        
        logger.info("Loaded SQL results", rows=len(documents))
        return documents


class DocumentLoader:
    """Factory for document loaders."""
    
    def __init__(self):
        self._loaders = {
            DocumentType.PDF: PDFLoader(),
            DocumentType.DOCX: DOCXLoader(),
            DocumentType.MARKDOWN: MarkdownLoader(),
            DocumentType.SQL: SQLLoader(),
        }
    
    def get_loader(self, doc_type: DocumentType) -> BaseLoader:
        """Get the appropriate loader for a document type."""
        if doc_type not in self._loaders:
            raise ValueError(f"Unsupported document type: {doc_type}")
        return self._loaders[doc_type]
    
    async def load_document(
        self,
        doc_type: DocumentType,
        source: str,
        metadata: dict[str, Any],
        **kwargs
    ) -> list[Document]:
        """Load a document using the appropriate loader."""
        loader = self.get_loader(doc_type)
        
        if doc_type == DocumentType.SQL:
            return await loader.load(
                connection_string=kwargs.get("connection_string", ""),
                query=source,
                metadata=metadata
            )
        elif doc_type in (DocumentType.PDF, DocumentType.DOCX):
            return await loader.load(file_path=source, metadata=metadata)
        else:
            return await loader.load(content=source, metadata=metadata)

import hashlib
import re
import uuid
from pathlib import Path
from typing import Protocol

import tiktoken
from dotenv import load_dotenv

from core.models import DocumentChunk

# Load environment variables
load_dotenv()

# Initialize OpenAI tokenizer for accurate token counting
ENCODING = tiktoken.encoding_for_model("gpt-4o")

# Legal document semantic separators (meaningful transitions)
LEGAL_SEPARATORS = [
    "\n\nARTICLE ",  # Contract articles
    "\n\nSection ",  # Contract sections
    "\n\nClause ",   # Contract clauses
    "\n\n##",       # Markdown headers
    "\n\nWHEREAS,", # Contract preamble
    "\n\nNOW THEREFORE,", # Contract operative clauses
    "\n\nIN WITNESS WHEREOF,", # Contract signatures
    "\n\n",         # Paragraph breaks
    "\n",           # Line breaks
    ".",             # Sentences
    " "              # Words
]


class DocumentParser(Protocol):
    def parse(self, file_path: Path) -> str: ...


class PDFParser:
    def parse(self, file_path: Path) -> str:
        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)


class TextParser:
    def parse(self, file_path: Path) -> str:
        return file_path.read_text(encoding="utf-8")


class SemanticChunker:
    """
    Industry best practice semantic chunker for legal documents.
    Splits based on meaningful document structure rather than fixed token limits.
    """
    
    def __init__(self):
        pass  # No configuration needed - pure semantic splitting
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens using OpenAI's tiktoken - same as GPT-4o."""
        return len(ENCODING.encode(text))
    
    def split_text(self, text: str) -> list[str]:
        """Split text using purely semantic boundaries based on legal structure."""
        # Split by major document sections using semantic patterns
        semantic_chunks = self._split_by_legal_structure(text)
        
        # If no semantic structure found, split by natural paragraph breaks
        if len(semantic_chunks) <= 1:
            semantic_chunks = self._split_by_paragraphs(text)
            
        # Filter out empty chunks
        return [chunk.strip() for chunk in semantic_chunks if chunk.strip()]
    
    def _split_by_legal_structure(self, text: str) -> list[str]:
        """Split by actual legal document structure (articles, sections, clauses)."""
        # Pattern 1: ARTICLE-based splitting (highest priority)
        article_pattern = r'\n(?=ARTICLE\s+[IVX\d]+)'
        article_splits = [chunk.strip() for chunk in re.split(article_pattern, text, flags=re.IGNORECASE) if chunk.strip()]
        if len(article_splits) > 1:
            return article_splits
        
        # Pattern 2: Section-based splitting  
        section_pattern = r'\n(?=Section\s+[IVX\d]+)'
        section_splits = [chunk.strip() for chunk in re.split(section_pattern, text, flags=re.IGNORECASE) if chunk.strip()]
        if len(section_splits) > 1:
            return section_splits
        
        # Pattern 3: Numbered clause splitting (1. 2. 3.)
        clause_pattern = r'\n(?=\d+\.\s+[A-Z])'
        clause_splits = [chunk.strip() for chunk in re.split(clause_pattern, text) if chunk.strip()]
        if len(clause_splits) > 1:
            return clause_splits
        
        # Pattern 4: Legal preamble sections
        preamble_pattern = r'\n(?=(?:WHEREAS,|NOW THEREFORE,|IN WITNESS WHEREOF,))'
        preamble_splits = [chunk.strip() for chunk in re.split(preamble_pattern, text, flags=re.IGNORECASE) if chunk.strip()]
        if len(preamble_splits) > 1:
            return preamble_splits
        
        # Pattern 5: Subsection patterns (a), (b), (c) or (i), (ii), (iii)
        subsection_pattern = r'\n(?=\([a-z]+\)\s+[A-Z])'
        subsection_splits = [chunk.strip() for chunk in re.split(subsection_pattern, text) if chunk.strip()]
        if len(subsection_splits) > 1:
            return subsection_splits
            
        # If no semantic structure found, return as single chunk
        return [text.strip()]
        
    def _split_by_paragraphs(self, text: str) -> list[str]:
        """Split by natural paragraph breaks when no legal structure is found."""
        # Split on double newlines (natural paragraph breaks)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # If still just one big chunk, split on single newlines
        if len(paragraphs) <= 1:
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
            
        return paragraphs
    
    def _identify_sections(self, text: str) -> list[str]:
        """Legacy method - kept for backward compatibility."""
        return self._split_by_legal_structure(text)


class MetadataEnricher:
    """Extracts structured metadata from legal document text."""

    def enrich(self, text: str, file_path: Path) -> dict:
        return {
            "doc_id": hashlib.md5(text.encode()).hexdigest()[:12],
            "filename": file_path.name,
            "doc_type": self._infer_doc_type(text),
            "parties": self._extract_parties(text),
            "date": self._extract_date(text),
            "client_id": self._extract_client_id(text),
            "jurisdiction": self._extract_jurisdiction(text),
            "clause_count": text.lower().count("clause"),
        }

    def _infer_doc_type(self, text: str) -> str:
        text_lower = text.lower()
        if "lease agreement" in text_lower:
            return "lease"
        if "non-disclosure" in text_lower or "nda" in text_lower:
            return "nda"
        if "employment" in text_lower:
            return "employment"
        if "settlement" in text_lower:
            return "settlement"
        return "contract"

    def _extract_parties(self, text: str) -> list[str]:
        pattern = r"(?:between|party[:\s]+|parties[:\s]+)([A-Z][A-Za-z\s,\.]+?)(?:and|,|\n)"
        matches = re.findall(pattern, text[:2000])
        return [m.strip() for m in matches[:3]]

    def _extract_date(self, text: str) -> str:
        pattern = r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+ \d{1,2},?\s?\d{4})\b"
        match = re.search(pattern, text[:1000])
        return match.group(1) if match else "unknown"
    
    def _extract_client_id(self, text: str) -> str:
        """Extract client identifier from legal documents."""
        # Look for common client ID patterns
        patterns = [
            r"Client\s+(?:ID|#)?:?\s*([A-Z0-9\-]+)",
            r"Matter\s+(?:No\.?|#)?:?\s*([A-Z0-9\-]+)",
            r"File\s+(?:No\.?|#)?:?\s*([A-Z0-9\-]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text[:2000], re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Fallback: extract first company name as identifier
        company_match = re.search(r"([A-Z][a-zA-Z\s]+(?:Inc|LLC|Corp|Ltd))", text[:1000])
        if company_match:
            return company_match.group(1).replace(" ", "_").upper()
        
        return "unknown"
    
    def _extract_jurisdiction(self, text: str) -> str:
        """Extract governing jurisdiction from legal documents."""
        # Look for governing law clauses
        patterns = [
            r"governed by the laws? of ([A-Z][a-zA-Z\s]+)",
            r"jurisdiction of ([A-Z][a-zA-Z\s]+)",
            r"courts? of ([A-Z][a-zA-Z\s]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                jurisdiction = match.group(1).strip()
                # Clean up common jurisdiction names
                if "california" in jurisdiction.lower():
                    return "California"
                elif "delaware" in jurisdiction.lower():
                    return "Delaware"
                elif "new york" in jurisdiction.lower():
                    return "New York"
                return jurisdiction
        
        return "unknown"


class IngestionPipeline:
    """Decoupled ingestion service — swap parsers or chunkers without touching retrieval."""

    PARSERS = {".pdf": PDFParser, ".txt": TextParser, ".md": TextParser}

    def __init__(self, use_semantic_chunking: bool = True):
        if use_semantic_chunking:
            self.splitter = SemanticChunker()
        else:
            # Fallback to standard chunking
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ".", " "],
            )
        self.enricher = MetadataEnricher()

    def ingest(self, file_path: Path) -> list[DocumentChunk]:
        suffix = file_path.suffix.lower()
        parser_cls = self.PARSERS.get(suffix, TextParser)
        raw_text = parser_cls().parse(file_path)

        metadata = self.enricher.enrich(raw_text, file_path)
        doc_id = metadata["doc_id"]

        raw_chunks = self.splitter.split_text(raw_text)

        return [
            DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                content=chunk,
                metadata={**metadata, "chunk_index": i, "total_chunks": len(raw_chunks)},
            )
            for i, chunk in enumerate(raw_chunks)
        ]

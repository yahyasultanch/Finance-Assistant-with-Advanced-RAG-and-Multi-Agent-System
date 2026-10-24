"""Text chunking utilities for RAG."""
from typing import List, Dict, Any, Tuple
import re
import logging

logger = logging.getLogger(__name__)


def contextual_header(metadata: Dict[str, Any]) -> str:
    """
    One-line context for a chunk: "NVDA 10-K (filed 2026-02-25) · Risk Factors".

    A chunk like "Competition could adversely impact our market share" says
    nothing about which company, filing or period it is from. Prepending this
    header before embedding and BM25 indexing lets queries match on that
    context (Anthropic's "contextual retrieval", with a metadata header
    instead of an LLM-written one).
    """
    # ponytail: metadata header only; an LLM-written per-chunk summary
    # (one call per chunk) would add more context if retrieval needs it
    ticker = metadata.get("ticker", "")
    form = metadata.get("form_type", "")
    date = metadata.get("filing_date", "")
    if form == "NEWS":
        return f"{ticker} news ({date}) · {metadata.get('source', '')}"
    section = metadata.get("section", "")
    section = "" if section in ("", "Document") else f" · {section}"
    return f"{ticker} {form} (filed {date}){section}"


def indexed_text(doc: Dict[str, Any]) -> str:
    """Text used for embedding, BM25 and reranking: header plus chunk."""
    context = doc.get("context", "")
    return f"{context}\n{doc['text']}" if context else doc["text"]


class DocumentChunker:
    """Chunks documents for RAG indexing."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(
        self, 
        text: str, 
        metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk text into overlapping segments.
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of chunk dicts with text and metadata
        """
        if metadata is None:
            metadata = {}
        
        # Clean text
        text = self._clean_text(text)
        
        # Split into sentences for better boundaries
        sentences = self._split_sentences(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "chunk_id": chunk_id,
                    "metadata": metadata.copy()
                })
                chunk_id += 1
                
                # Start new chunk with overlap
                overlap_text = chunk_text[-self.chunk_overlap:] if len(chunk_text) > self.chunk_overlap else chunk_text
                overlap_sentences = self._split_sentences(overlap_text)
                current_chunk = overlap_sentences
                current_size = len(overlap_text)
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "chunk_id": chunk_id,
                "metadata": metadata.copy()
            })
        
        logger.info(f"Created {len(chunks)} chunks from text of length {len(text)}")
        return chunks

    def chunk_filing(
        self, 
        filing_text: str,
        filing_metadata: Dict[str, Any],
        sections: Dict[str, str] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk SEC filing with section-aware splitting.
        
        Args:
            filing_text: Full filing text
            filing_metadata: Filing metadata (form type, date, etc.)
            sections: Dict of section_name -> section_text (optional)
            
        Returns:
            List of chunks with section metadata
        """
        all_chunks = []
        
        if sections:
            # Chunk each section separately
            for section_name, section_text in sections.items():
                metadata = filing_metadata.copy()
                metadata["section"] = section_name
                
                section_chunks = self.chunk_text(section_text, metadata)
                all_chunks.extend(section_chunks)
        else:
            # Chunk entire filing
            all_chunks = self.chunk_text(filing_text, filing_metadata)
        
        return all_chunks

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,;:!?()\-\'\"$%]', '', text)
        return text.strip()

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitter
        sentence_endings = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_endings.split(text)
        return [s.strip() for s in sentences if s.strip()]

    # Section name -> (header regex, regexes of headers that end the section)
    TEN_K_SECTIONS = {
        "Business": (r"Item\s+1\.\s*Business", [r"Item\s+1[AB]", r"Item\s+2"]),
        "Risk Factors": (r"Item\s+1A\.\s*Risk Factors", [r"Item\s+1B", r"Item\s+2"]),
        "MD&A": (r"Item\s+7\.\s*Management.?s Discussion", [r"Item\s+7A", r"Item\s+8"]),
        "Financial Statements": (r"Item\s+8\.\s*Financial Statements", [r"Item\s+9"]),
    }

    TEN_Q_SECTIONS = {
        "Financial Statements": (r"Item\s+1\.\s*Financial Statements", [r"Item\s+2"]),
        "MD&A": (r"Item\s+2\.\s*Management.?s Discussion", [r"Item\s+3"]),
        "Risk Factors": (r"Item\s+1A\.\s*Risk Factors", [r"Item\s+2"]),
    }

    def extract_sections_from_10k(self, filing_text: str) -> Dict[str, str]:
        """
        Extract standard sections from 10-K filing.
        
        Args:
            filing_text: Full 10-K text
            
        Returns:
            Dict of section_name -> section_text
        """
        return self._extract_sections(filing_text, self.TEN_K_SECTIONS)

    def extract_sections_from_10q(self, filing_text: str) -> Dict[str, str]:
        """
        Extract standard sections from 10-Q filing.
        
        Args:
            filing_text: Full 10-Q text
            
        Returns:
            Dict of section_name -> section_text
        """
        return self._extract_sections(filing_text, self.TEN_Q_SECTIONS)

    def _extract_sections(
        self,
        filing_text: str,
        specs: Dict[str, Tuple[str, List[str]]]
    ) -> Dict[str, str]:
        """Extract each (header, end headers) spec, keeping substantial sections."""
        sections = {}
        for section_name, (header, ends) in specs.items():
            section_text = self._longest_match(header, ends, filing_text)
            if len(section_text) > 100:  # Only include substantial sections
                sections[section_name] = section_text
                logger.info(f"Extracted section: {section_name} ({len(section_text)} chars)")
        return sections

    @staticmethod
    def _longest_match(header: str, ends: List[str], text: str) -> str:
        """
        Return the longest span from a section header to the next end header.
        
        Filings repeat each "Item X." header in the table of contents and in
        cross-references ("Refer to Item 1A. Risk Factors..."). Real headers
        start a line, so first try line-anchored headers, ending at the next
        line-anchored "Item N" of any kind; TOC entries then end almost
        immediately and the longest match is the real section. Falls back to
        matching anywhere, with the given end headers, for filings whose text
        layout puts headers mid-line.
        """
        line_start = r"^[^\S\n]*"
        passes = [
            (line_start + header, line_start + r"Item\s+\d+[A-Z]?\b"),
            (header, "|".join(ends)),
        ]
        for start, end in passes:
            pattern = rf"{start}(.*?)(?={end}|\Z)"
            matches = [
                m.group(1).strip()
                for m in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE | re.MULTILINE)
            ]
            best = max(matches, key=len, default="")
            if len(best) > 100:
                return best
        return ""

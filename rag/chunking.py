"""Text chunking utilities for RAG."""
from typing import List, Dict, Any
import re
import logging

logger = logging.getLogger(__name__)


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

    def extract_sections_from_10k(self, filing_text: str) -> Dict[str, str]:
        """
        Extract standard sections from 10-K filing.
        
        Args:
            filing_text: Full 10-K text
            
        Returns:
            Dict of section_name -> section_text
        """
        sections = {}
        
        # Common 10-K sections
        section_patterns = {
            "Business": r"(?:ITEM\s+1\.|Item\s+1\.)(?:\s*Business)(.*?)(?=ITEM\s+1[AB]|Item\s+1[AB]|ITEM\s+2|Item\s+2|\Z)",
            "Risk Factors": r"(?:ITEM\s+1A\.|Item\s+1A\.)(?:\s*Risk Factors)(.*?)(?=ITEM\s+1B|Item\s+1B|ITEM\s+2|Item\s+2|\Z)",
            "MD&A": r"(?:ITEM\s+7\.|Item\s+7\.)(?:\s*Management'?s Discussion)(.*?)(?=ITEM\s+7A|Item\s+7A|ITEM\s+8|Item\s+8|\Z)",
            "Financial Statements": r"(?:ITEM\s+8\.|Item\s+8\.)(?:\s*Financial Statements)(.*?)(?=ITEM\s+9|Item\s+9|\Z)",
        }
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, filing_text, re.DOTALL | re.IGNORECASE)
            if match:
                section_text = match.group(1).strip()
                if len(section_text) > 100:  # Only include substantial sections
                    sections[section_name] = section_text
                    logger.info(f"Extracted section: {section_name} ({len(section_text)} chars)")
        
        return sections

    def extract_sections_from_10q(self, filing_text: str) -> Dict[str, str]:
        """
        Extract standard sections from 10-Q filing.
        
        Args:
            filing_text: Full 10-Q text
            
        Returns:
            Dict of section_name -> section_text
        """
        sections = {}
        
        section_patterns = {
            "Financial Statements": r"(?:ITEM\s+1\.|Item\s+1\.)(?:\s*Financial Statements)(.*?)(?=ITEM\s+2|Item\s+2|\Z)",
            "MD&A": r"(?:ITEM\s+2\.|Item\s+2\.)(?:\s*Management'?s Discussion)(.*?)(?=ITEM\s+3|Item\s+3|\Z)",
            "Risk Factors": r"(?:ITEM\s+1A\.|Item\s+1A\.)(?:\s*Risk Factors)(.*?)(?=ITEM\s+2|Item\s+2|\Z)",
        }
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, filing_text, re.DOTALL | re.IGNORECASE)
            if match:
                section_text = match.group(1).strip()
                if len(section_text) > 100:
                    sections[section_name] = section_text
                    logger.info(f"Extracted section: {section_name} ({len(section_text)} chars)")
        
        return sections

"""Summarizer agent with RAG and citation tracking."""
from openai import AzureOpenAI
from typing import List, Dict, Any, Tuple
import logging
import re

logger = logging.getLogger(__name__)


class SummarizerAgent:
    """Agent for generating grounded summaries with citations."""

    def __init__(
        self,
        retriever,
        azure_endpoint: str,
        azure_key: str,
        azure_deployment: str,
        api_version: str = "2024-12-01-preview",
        top_k: int = 8
    ):
        """
        Initialize summarizer agent.
        
        Args:
            retriever: HybridRetriever instance
            azure_endpoint: Azure OpenAI endpoint URL
            azure_key: Azure OpenAI API key
            azure_deployment: Deployment name (e.g., gpt-4o-mini)
            api_version: API version
            top_k: Number of chunks to retrieve
        """
        self.retriever = retriever
        self.azure_endpoint = azure_endpoint
        self.azure_deployment = azure_deployment
        self.top_k = top_k
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
            api_version=api_version
        )

    def generate_summary(
        self,
        query: str,
        ticker: str,
        max_bullets: int = 3
    ) -> Dict[str, Any]:
        """
        Generate a grounded summary with citations.
        
        Args:
            query: Query to answer
            ticker: Stock ticker
            max_bullets: Maximum number of bullet points
            
        Returns:
            Dict with summary text, citations, and retrieved contexts
        """
        # Retrieve relevant contexts
        results = self.retriever.retrieve(query, k=self.top_k)
        
        if not results:
            return {
                "summary": "No relevant information found in filings.",
                "citations": [],
                "contexts": []
            }
        
        # Build context string with citations
        contexts = []
        citation_map = {}
        
        for i, (doc, score) in enumerate(results):
            citation_id = i + 1
            text = doc["text"][:1000]  # Limit context length
            metadata = doc.get("metadata", {})
            
            source = f"{metadata.get('form_type', 'N/A')} ({metadata.get('filing_date', 'N/A')})"
            section = metadata.get('section', 'Document')
            
            citation_ref = f"[{citation_id}:{metadata.get('form_type', 'Doc')}#{section}]"
            citation_map[citation_id] = {
                "reference": citation_ref,
                "source": source,
                "section": section,
                "url": metadata.get('url', ''),
                "date": metadata.get('filing_date', '')
            }
            
            contexts.append({
                "id": citation_id,
                "text": text,
                "citation": citation_ref,
                "metadata": metadata
            })
        
        # Build context for LLM
        context_text = "\n\n".join([
            f"Context {ctx['id']}: {ctx['text']}\nSource: {ctx['citation']}"
            for ctx in contexts
        ])
        
        # Generate summary with Azure OpenAI
        system_prompt = f"""You are an equity research assistant analyzing {ticker}. 
        
Your task is to answer the user's question using ONLY the provided context from SEC filings and news.

CRITICAL RULES:
1. Only use information explicitly stated in the contexts
2. Cite sources using the format [1:10-K#Business], [2:10-Q#MD&A], etc.
3. Provide exactly {max_bullets} concise bullet points (unless less information available)
4. Each bullet must end with at least one citation
5. If information is not in the context, say "Not found in filings"
6. Be specific and quote-backed - use numbers, dates, and facts from filings

Format your response as bullet points with citations at the end of each point."""

        user_prompt = f"""Question: {query}

Available Context:
{context_text}

Provide {max_bullets} bullet points that directly answer the question, with citations."""

        summary_text = self._call_ollama(system_prompt, user_prompt)
        
        # Extract citations from summary
        cited_ids = self._extract_citation_ids(summary_text)
        citations = [citation_map[cid] for cid in cited_ids if cid in citation_map]
        
        logger.info(f"Extracted {len(cited_ids)} citation IDs from summary: {cited_ids}")
        logger.info(f"Matched {len(citations)} citations from citation_map")
        
        if len(cited_ids) > 0 and len(citations) == 0:
            logger.warning(f"Citation IDs found but no matches in citation_map. IDs: {cited_ids}, Map keys: {list(citation_map.keys())}")
        
        return {
            "summary": summary_text,
            "citations": citations,
            "contexts": contexts,
            "query": query
        }

    def generate_multiple_summaries(
        self,
        queries: List[str],
        ticker: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate summaries for multiple queries.
        
        Args:
            queries: List of queries
            ticker: Stock ticker
            
        Returns:
            Dict mapping query to summary result
        """
        results = {}
        
        for query in queries:
            logger.info(f"Generating summary for: {query}")
            results[query] = self.generate_summary(query, ticker)
        
        return results

    def get_company_overview(self, ticker: str) -> Dict[str, Any]:
        """Get company overview summary."""
        return self.generate_summary(
            f"Provide a concise overview of {ticker}'s business model and operations",
            ticker,
            max_bullets=3
        )

    def get_growth_drivers(self, ticker: str) -> Dict[str, Any]:
        """Get key growth drivers."""
        return self.generate_summary(
            f"What are the key growth drivers and opportunities for {ticker}?",
            ticker,
            max_bullets=3
        )

    def get_risk_factors(self, ticker: str) -> Dict[str, Any]:
        """Get top risk factors."""
        return self.generate_summary(
            f"What are the top 3 most significant risks facing {ticker}?",
            ticker,
            max_bullets=3
        )

    def get_recent_events(self, ticker: str) -> Dict[str, Any]:
        """Get recent material events from 8-K filings."""
        return self.generate_summary(
            f"What are the most recent material events, announcements, or significant developments for {ticker}? Include earnings releases, executive changes, acquisitions, agreements, shareholder meetings, or other important corporate events.",
            ticker,
            max_bullets=5  # Increase to 5 for events
        )

    def get_financial_outlook(self, ticker: str) -> Dict[str, Any]:
        """Get financial outlook or guidance."""
        return self.generate_summary(
            f"What is {ticker}'s financial outlook, guidance, or future expectations?",
            ticker,
            max_bullets=3
        )

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call Azure OpenAI API for text generation.
        
        Args:
            system_prompt: System instruction
            user_prompt: User query
            
        Returns:
            Generated text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.azure_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower temperature for factual responses
                max_tokens=500,
                top_p=0.9
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error calling Azure OpenAI: {e}")
            return f"Error generating summary: {str(e)}"

    def _extract_citation_ids(self, text: str) -> List[int]:
        """
        Extract citation IDs from text.
        
        Args:
            text: Text with citations like [1:10-K#Business]
            
        Returns:
            List of citation IDs
        """
        pattern = r'\[(\d+):[^\]]+\]'
        matches = re.findall(pattern, text)
        return [int(m) for m in matches]

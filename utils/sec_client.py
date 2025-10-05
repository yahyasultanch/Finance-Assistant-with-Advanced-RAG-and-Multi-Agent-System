"""SEC EDGAR client for fetching company filings."""
import requests
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class SECClient:
    """Client for accessing SEC EDGAR data."""

    def __init__(self, user_agent: str, cache_dir: Path):
        """
        Initialize SEC client.
        
        Args:
            user_agent: User agent string (required by SEC)
            cache_dir: Directory to cache downloaded filings
        """
        self.user_agent = user_agent
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://data.sec.gov"
        self.headers = {"User-Agent": user_agent}
        self.rate_limit_delay = 0.1  # SEC requires 10 requests/second max

    def _make_request(self, url: str) -> Optional[Dict]:
        """Make rate-limited request to SEC API."""
        time.sleep(self.rate_limit_delay)
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def get_cik(self, ticker: str) -> Optional[str]:
        """
        Get CIK (Central Index Key) for a ticker symbol.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            CIK as zero-padded string or None
        """
        ticker = ticker.upper()
        url = f"{self.base_url}/submissions/CIK{ticker}.json"
        
        # Try direct lookup
        data = self._make_request(url)
        if data:
            return data.get("cik")
        
        # Fallback: search ticker mapping
        try:
            tickers_url = "https://www.sec.gov/files/company_tickers.json"
            time.sleep(self.rate_limit_delay)
            response = requests.get(tickers_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            tickers_data = response.json()
            
            for entry in tickers_data.values():
                if entry.get("ticker", "").upper() == ticker:
                    cik = str(entry["cik_str"]).zfill(10)
                    return cik
        except Exception as e:
            logger.error(f"Error looking up CIK for {ticker}: {e}")
        
        return None

    def get_company_submissions(self, cik: str) -> Optional[Dict]:
        """
        Get all submissions for a company.
        
        Args:
            cik: Company CIK (zero-padded)
            
        Returns:
            Submissions data or None
        """
        url = f"{self.base_url}/submissions/CIK{cik}.json"
        return self._make_request(url)

    def get_recent_filings(
        self, 
        ticker: str, 
        form_types: List[str] = ["10-K", "10-Q", "8-K"],
        days_back: int = 365
    ) -> List[Dict[str, Any]]:
        """
        Get recent filings for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            form_types: List of form types to retrieve
            days_back: Number of days to look back
            
        Returns:
            List of filing metadata dicts
        """
        cik = self.get_cik(ticker)
        if not cik:
            logger.error(f"Could not find CIK for ticker {ticker}")
            return []
        
        submissions = self.get_company_submissions(cik)
        if not submissions:
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        filings = []
        
        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        accession_numbers = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        
        for i, form in enumerate(forms):
            if form in form_types:
                filing_date_str = filing_dates[i]
                filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d")
                
                if filing_date >= cutoff_date:
                    accession = accession_numbers[i].replace("-", "")
                    primary_doc = primary_docs[i]
                    
                    filings.append({
                        "form_type": form,
                        "filing_date": filing_date_str,
                        "accession_number": accession_numbers[i],
                        "cik": cik,
                        "primary_document": primary_doc,
                        "document_url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_doc}"
                    })
        
        return sorted(filings, key=lambda x: x["filing_date"], reverse=True)

    def download_filing(self, filing: Dict[str, Any]) -> Optional[str]:
        """
        Download filing document.
        
        Args:
            filing: Filing metadata dict
            
        Returns:
            Path to downloaded file or None
        """
        url = filing["document_url"]
        filename = f"{filing['form_type']}_{filing['filing_date']}_{filing['accession_number']}.html"
        filepath = self.cache_dir / filename
        
        # Return cached file if exists
        if filepath.exists():
            logger.info(f"Using cached filing: {filepath}")
            return str(filepath)
        
        try:
            time.sleep(self.rate_limit_delay)
            response = requests.get(url, headers=self.headers, timeout=60)
            response.raise_for_status()
            
            filepath.write_bytes(response.content)
            logger.info(f"Downloaded filing to: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Error downloading filing from {url}: {e}")
            return None

    def get_company_facts(self, cik: str) -> Optional[Dict]:
        """
        Get company facts (XBRL data).
        
        Args:
            cik: Company CIK
            
        Returns:
            Company facts data or None
        """
        url = f"{self.base_url}/api/xbrl/companyfacts/CIK{cik}.json"
        return self._make_request(url)

    def parse_filing_text(self, filepath: str) -> str:
        """
        Extract text from HTML filing.
        
        Args:
            filepath: Path to HTML filing
            
        Returns:
            Extracted text
        """
        try:
            from bs4 import BeautifulSoup
            
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text(separator='\n')
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            logger.error(f"Error parsing filing {filepath}: {e}")
            return ""

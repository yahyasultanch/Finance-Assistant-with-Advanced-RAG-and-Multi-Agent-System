"""SEC EDGAR client for fetching company filings."""
import requests
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
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

    # 8-K Item mapping: https://www.sec.gov/files/form8-k.pdf
    EIGHT_K_ITEMS = {
        "1.01": "Material Agreement",
        "1.02": "Termination of Material Agreement",
        "1.03": "Bankruptcy or Receivership",
        "1.04": "Mine Safety Disclosure",
        "2.01": "Completion of Acquisition or Disposition",
        "2.02": "Results of Operations and Financial Condition",
        "2.03": "Creation of Direct Financial Obligation",
        "2.04": "Triggering Events That Accelerate Obligations",
        "2.05": "Costs Associated with Exit Activities",
        "2.06": "Material Impairments",
        "3.01": "Notice of Delisting",
        "3.02": "Unregistered Sales of Equity Securities",
        "3.03": "Material Modifications to Rights of Security Holders",
        "4.01": "Changes in Registrant's Certifying Accountant",
        "4.02": "Non-Reliance on Previously Issued Financial Statements",
        "5.01": "Changes in Control of Registrant",
        "5.02": "Departure/Election of Directors or Officers",
        "5.03": "Amendments to Articles of Incorporation or Bylaws",
        "5.04": "Temporary Suspension of Trading",
        "5.05": "Amendments to Registrant's Code of Ethics",
        "5.06": "Change in Shell Company Status",
        "5.07": "Submission of Matters to Vote of Security Holders",
        "5.08": "Shareholder Nominations",
        "7.01": "Regulation FD Disclosure",
        "8.01": "Other Events",
        "9.01": "Financial Statements and Exhibits"
    }

    def parse_8k_items(self, filepath: str) -> List[Dict[str, str]]:
        """
        Parse 8-K filing to extract Item numbers and descriptions.
        
        Args:
            filepath: Path to 8-K HTML filing
            
        Returns:
            List of dicts with 'item', 'item_name', 'description'
        """
        try:
            from bs4 import BeautifulSoup
            
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'lxml')
            text = soup.get_text()
            
            # Find Item patterns like "Item 2.02", "Item 5.02", etc.
            item_pattern = r'Item\s+(\d+\.\d+)[:\.\s]+([^\n]+)'
            matches = re.finditer(item_pattern, text, re.IGNORECASE)
            
            items_found = []
            for match in matches:
                item_num = match.group(1)
                item_desc = match.group(2).strip()
                
                # Get the standard item name
                item_name = self.EIGHT_K_ITEMS.get(item_num, "Other")
                
                # Extract more context (next 500 chars after item header)
                start_pos = match.end()
                context = text[start_pos:start_pos + 500].strip()
                
                # Clean up context
                context = ' '.join(context.split())[:300]
                
                items_found.append({
                    "item": item_num,
                    "item_name": item_name,
                    "description": context if context else item_desc
                })
            
            # Remove duplicates (keep first occurrence)
            seen = set()
            unique_items = []
            for item in items_found:
                if item['item'] not in seen:
                    seen.add(item['item'])
                    unique_items.append(item)
            
            return unique_items
            
        except Exception as e:
            logger.error(f"Error parsing 8-K items from {filepath}: {e}")
            return []

    def extract_8k_event_summary(self, filing: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of an 8-K event.
        
        Args:
            filing: Filing metadata with 'filepath' key
            
        Returns:
            Event summary string
        """
        filepath = filing.get('filepath')
        if not filepath:
            return f"8-K Filing on {filing.get('filing_date', 'unknown date')}"
        
        items = self.parse_8k_items(filepath)
        
        if not items:
            return f"8-K Filing on {filing.get('filing_date', 'unknown date')}"
        
        # Create summary based on most important items
        primary_items = [item for item in items if item['item'] in ['2.02', '1.01', '5.02', '2.01']]
        
        if not primary_items:
            primary_items = items[:2]  # Take first 2 items if no priority items
        
        if len(primary_items) == 1:
            item = primary_items[0]
            return f"{item['item_name']}: {item['description'][:200]}"
        else:
            item_names = [item['item_name'] for item in primary_items]
            return f"Multiple events: {', '.join(item_names)}"

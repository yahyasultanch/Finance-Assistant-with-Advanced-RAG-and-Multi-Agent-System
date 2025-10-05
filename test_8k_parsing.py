"""Test 8-K parsing functionality."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.sec_client import SECClient
from config import SEC_USER_AGENT, DATA_DIR

def test_8k_parsing():
    """Test 8-K item extraction."""
    print("Testing 8-K Parsing...")
    print("=" * 60)
    
    # Initialize SEC client
    sec_client = SECClient(SEC_USER_AGENT, DATA_DIR / "raw")
    
    # Get recent NVDA 8-K filings
    print("\n1. Fetching NVDA 8-K filings...")
    filings = sec_client.get_recent_filings("NVDA", form_types=["8-K"], days_back=365)
    print(f"   Found {len(filings)} 8-K filings")
    
    if not filings:
        print("   ❌ No 8-K filings found")
        return
    
    # Test parsing first 3 filings
    print(f"\n2. Parsing first 3 8-K filings...")
    for i, filing in enumerate(filings[:3], 1):
        print(f"\n   Filing {i}: {filing['filing_date']}")
        print(f"   URL: {filing['document_url']}")
        
        # Download filing
        filepath = sec_client.download_filing(filing)
        if not filepath:
            print(f"   ❌ Could not download filing")
            continue
        
        filing['filepath'] = filepath
        print(f"   ✅ Downloaded to: {filepath}")
        
        # Parse items
        items = sec_client.parse_8k_items(filepath)
        print(f"   Found {len(items)} items:")
        
        for item in items:
            print(f"      - Item {item['item']}: {item['item_name']}")
            print(f"        {item['description'][:150]}...")
        
        # Generate event summary
        event_summary = sec_client.extract_8k_event_summary(filing)
        print(f"\n   📋 Event Summary:")
        print(f"      {event_summary}")
        print()
    
    print("\n" + "=" * 60)
    print("✅ 8-K Parsing Test Complete!")

if __name__ == "__main__":
    test_8k_parsing()

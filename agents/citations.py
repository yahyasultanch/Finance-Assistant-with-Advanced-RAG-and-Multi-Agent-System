"""Source registry and claim verification for grounded reports.

Every tool result is turned into sources with stable ids. Ids are content
hashes, so parallel agents never collide and their source maps merge cleanly.
Claims cite source ids, and checks run on each claim:
  1. every cited id exists
  2. every number in the claim appears in a cited source (allowing for unit
     scaling: "$46.7 billion" matches "46,743,000,000")
  3. an LLM judge confirms the sources support the claim (in the graph's
     verifier, since it needs the model)
"""
import hashlib
import re
from typing import Any, Dict, Iterable, List
from urllib.parse import quote

# Lower is more trustworthy
TIERS = {
    "sec": 1,          # SEC filings and XBRL facts
    "market_data": 2,  # prices and company info
    "news": 3,         # publishers
    "social": 4,       # Reddit etc.
    "memory": 5,       # this system's own earlier briefs
}

ID_PATTERN = re.compile(r"\bS[0-9a-f]{6}\b")
NUMBER_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?")


def make_source(
    kind: str,
    label: str,
    title: str,
    url: str,
    date: str,
    text: str,
    tier: str = "sec",
) -> Dict[str, Any]:
    """
    Build a source with a content-hash id.

    Args:
        kind: filing, xbrl, event, news or market_data
        label: Short citation label, e.g. "10-K#Risk Factors" or "XBRL#Revenue"
        title: Human readable title for the appendix
        url: Link to the original document
        date: Filing or publish date
        text: Exact text the claim can be checked against
        tier: Key into TIERS
    """
    digest = hashlib.sha1(f"{url}|{label}|{text}".encode("utf-8")).hexdigest()
    return {
        "id": f"S{digest[:6]}",
        "kind": kind,
        "label": label,
        "title": title,
        "url": url,
        "date": date,
        "text": text,
        "tier": TIERS[tier],
    }


def render_sources(sources: Iterable[Dict[str, Any]], max_chars: int = 1200) -> str:
    """Format sources for a tool result: id header line, then the text."""
    blocks = [
        f"[{s['id']}] {s['label']} ({s['date']}) - {s['title']}\n{s['text'][:max_chars]}"
        for s in sources
    ]
    return "\n\n".join(blocks) if blocks else "No results."


def deep_link(source: Dict[str, Any], words: int = 6) -> str:
    """
    Link with a text fragment (#:~:text=) so browsers scroll to the quote.
    Uses the first run of plain words, which survive HTML-to-text cleanup.
    """
    url = source.get("url", "")
    # Only filing chunks quote the page verbatim; event and XBRL texts are composed
    if not url or source.get("kind") != "filing":
        return url
    plain = re.findall(r"[A-Za-z][A-Za-z']+", source.get("text", ""))
    if len(plain) < words:
        return url
    return f"{url}#:~:text={quote(' '.join(plain[:words]))}"


def numbers_in(text: str) -> List[float]:
    """
    Numbers a claim asserts, skipping years and small counts.

    "Revenue rose 56% to $46.7 billion in fiscal 2025" -> [56.0, 46.7]
    """
    values = []
    for token in NUMBER_PATTERN.findall(text):
        value = float(token.replace(",", ""))
        is_int = "." not in token
        # ponytail: skips years and integers under 10 ("3 segments"); a claim
        # whose only number is like that relies on the LLM judge alone
        if is_int and (value < 10 or 1990 <= value <= 2100):
            continue
        values.append(value)
    return values


def number_supported(value: float, source_numbers: List[float], tolerance: float = 0.01) -> bool:
    """True if value matches a source number within 1%, at any of 1/1e3/1e6/1e9 scale."""
    for source_value in source_numbers:
        for scale in (1, 1e3, 1e6, 1e9):
            if abs(value - source_value / scale) <= tolerance * value:
                return True
    return False


def check_claim(claim: Dict[str, Any], sources: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Deterministic checks for one claim: citations exist, numbers appear.

    Args:
        claim: {"text": ..., "sources": [ids]}
        sources: Registry id -> source

    Returns:
        Claim with "status" (ok, bad_citation, number_mismatch) and "reason"
    """
    cited = [sid for sid in claim.get("sources", []) if sid in sources]
    unknown = [sid for sid in claim.get("sources", []) if sid not in sources]
    result = {**claim, "sources": cited}

    if not cited:
        return {**result, "status": "bad_citation",
                "reason": f"No valid source ids (got {claim.get('sources', [])})"}

    source_numbers = [abs(n) for sid in cited for n in numbers_in_raw(sources[sid]["text"])]
    missing = [n for n in numbers_in(claim["text"]) if not number_supported(n, source_numbers)]
    if missing:
        return {**result, "status": "number_mismatch",
                "reason": f"Numbers not found in cited sources: {missing}"}

    reason = f"Ignored unknown ids {unknown}" if unknown else ""
    return {**result, "status": "ok", "reason": reason}


def numbers_in_raw(text: str) -> List[float]:
    """Every number in a source text (no filtering)."""
    return [float(t.replace(",", "")) for t in NUMBER_PATTERN.findall(text)]

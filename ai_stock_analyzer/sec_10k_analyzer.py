"""
SEC 10-K Filing Fetcher and Analyzer.

Fetches the most recent 10-K filing for a given ticker symbol from the
SEC EDGAR API (no API key required) and extracts key sections:
  - Business Overview (Item 1)
  - Risk Factors (Item 1A)
  - Management Discussion & Analysis (Item 7)
  - Financial Highlights

Usage
-----
    from sec_10k_analyzer import SEC10KAnalyzer
    analyzer = SEC10KAnalyzer()
    report = analyzer.get_10k_summary("MU")
    print(report)
"""

import re
import time
import textwrap
from typing import Optional

import requests
from bs4 import BeautifulSoup


EDGAR_HEADERS = {
    "User-Agent": "AI-Stock-Analyzer research@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

EDGAR_FULL_HEADERS = {
    "User-Agent": "AI-Stock-Analyzer research@example.com",
    "Accept-Encoding": "gzip, deflate",
}

SECTION_PATTERNS = {
    "Business Overview": [
        r"item\s*1[.\s]+business",
        r"part\s*i.*item\s*1",
    ],
    "Risk Factors": [
        r"item\s*1a[.\s]+risk\s*factors",
        r"risk\s*factors",
    ],
    "MD&A": [
        r"item\s*7[.\s]+management",
        r"management.{0,20}discussion",
    ],
}

MAX_SECTION_CHARS = 3000  # characters per section in summary


class SEC10KAnalyzer:
    """Fetches and parses SEC 10-K annual reports."""

    BASE_EDGAR = "https://data.sec.gov"
    COMPANY_SEARCH = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt={year}-01-01&enddt={year}-12-31&forms=10-K"
    CIK_LOOKUP = "https://www.sec.gov/cgi-bin/browse-edgar?company=&CIK={ticker}&type=10-K&dateb=&owner=include&count=5&search_text=&action=getcompany"

    def __init__(self, cache: bool = True):
        self._cache: dict = {}
        self._use_cache = cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_10k_summary(self, ticker: str, year: Optional[int] = None) -> str:
        """
        Return a formatted 10-K summary for *ticker*.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol (e.g. "MU", "AAPL").
        year : int, optional
            Fiscal year for the filing. Defaults to the most recent filing.

        Returns
        -------
        str
            Multi-section formatted report suitable for display.
        """
        ticker = ticker.upper().strip()
        cache_key = f"{ticker}:{year}"
        if self._use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        cik = self._get_cik(ticker)
        if not cik:
            return self._error_report(ticker, "Could not find CIK for this ticker on SEC EDGAR.")

        filing_url, filing_date = self._get_latest_10k_url(cik, year)
        if not filing_url:
            return self._error_report(ticker, "No 10-K filing found on SEC EDGAR.")

        text = self._fetch_filing_text(filing_url)
        if not text:
            return self._error_report(ticker, "Could not retrieve filing document.")

        report = self._build_report(ticker, filing_date, text)
        if self._use_cache:
            self._cache[cache_key] = report
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_cik(self, ticker: str) -> Optional[str]:
        """Return zero-padded CIK for a ticker using the EDGAR company API."""
        try:
            url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company=&CIK={ticker}&type=10-K&dateb=&owner=include&count=10&search_text=&output=atom"
            resp = requests.get(url, headers=EDGAR_FULL_HEADERS, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            # Look for the CIK in the atom feed
            cik_tag = soup.find("cik")
            if cik_tag:
                return cik_tag.text.strip().zfill(10)

            # Fallback: parse from the HTML redirect URL
            cik_tag = soup.find("companyname")
            link = soup.find("filing-href")
            if link:
                match = re.search(r"CIK=(\d+)", link.text)
                if match:
                    return match.group(1).zfill(10)
        except Exception:
            pass

        # Second approach: company_facts JSON endpoint
        try:
            tickers_url = "https://www.sec.gov/files/company_tickers.json"
            resp = requests.get(tickers_url, headers=EDGAR_FULL_HEADERS, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker:
                    return str(entry["cik_str"]).zfill(10)
        except Exception:
            pass

        return None

    def _get_latest_10k_url(self, cik: str, year: Optional[int]) -> tuple:
        """Return (document_url, filing_date) for the most recent 10-K."""
        try:
            submissions_url = f"{self.BASE_EDGAR}/submissions/CIK{cik}.json"
            resp = requests.get(submissions_url, headers=EDGAR_HEADERS, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            filings = data.get("filings", {}).get("recent", {})
            forms = filings.get("form", [])
            dates = filings.get("filingDate", [])
            accession_numbers = filings.get("accessionNumber", [])

            for form, date, accession in zip(forms, dates, accession_numbers):
                if form != "10-K":
                    continue
                if year and not date.startswith(str(year)):
                    continue
                acc_clean = accession.replace("-", "")
                filing_index_url = (
                    f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{accession}-index.htm"
                )
                doc_url = self._find_10k_document(filing_index_url, int(cik), acc_clean)
                if doc_url:
                    return doc_url, date
        except Exception:
            pass
        return None, None

    def _find_10k_document(self, index_url: str, cik: int, acc_clean: str) -> Optional[str]:
        """Return the URL of the primary 10-K document from a filing index page."""
        try:
            resp = requests.get(index_url, headers=EDGAR_FULL_HEADERS, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            for row in soup.find_all("tr"):
                cols = row.find_all("td")
                if len(cols) >= 4:
                    doc_type = cols[3].text.strip() if len(cols) > 3 else ""
                    if "10-K" in doc_type and "EX" not in doc_type:
                        link = cols[2].find("a") if len(cols) > 2 else None
                        if link and link.get("href"):
                            href = link["href"]
                            if href.startswith("/"):
                                return f"https://www.sec.gov{href}"
                            return href
            # Fallback: look for any .htm link
            for link in soup.find_all("a"):
                href = link.get("href", "")
                if href.endswith(".htm") and acc_clean.lower() in href.lower():
                    if href.startswith("/"):
                        return f"https://www.sec.gov{href}"
                    return href
        except Exception:
            pass
        return None

    def _fetch_filing_text(self, url: str) -> Optional[str]:
        """Fetch and return the plain text of a filing."""
        try:
            time.sleep(0.5)  # be polite to SEC servers
            resp = requests.get(url, headers=EDGAR_FULL_HEADERS, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            return soup.get_text(separator="\n")
        except Exception:
            return None

    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract a section from the filing text using regex patterns."""
        patterns = SECTION_PATTERNS.get(section_name, [])
        lines = text.split("\n")
        start_idx = None

        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            for pattern in patterns:
                if re.search(pattern, line_lower):
                    start_idx = i
                    break
            if start_idx is not None:
                break

        if start_idx is None:
            return "(Section not found in filing)"

        # Collect up to MAX_SECTION_CHARS worth of text
        collected = []
        char_count = 0
        for line in lines[start_idx + 1:]:
            if char_count >= MAX_SECTION_CHARS:
                break
            stripped = line.strip()
            if stripped:
                collected.append(stripped)
                char_count += len(stripped)

        raw = " ".join(collected)
        # Wrap to 80 chars for clean display
        return textwrap.fill(raw[:MAX_SECTION_CHARS], width=80)

    def _build_report(self, ticker: str, filing_date: str, text: str) -> str:
        """Build the formatted report string."""
        lines = []
        border = "=" * 80
        thin = "-" * 80

        lines.append(border)
        lines.append(f"  SEC 10-K ANNUAL REPORT ANALYSIS  —  {ticker}")
        lines.append(f"  Filing Date: {filing_date}")
        lines.append(border)

        for section_name in ["Business Overview", "Risk Factors", "MD&A"]:
            lines.append(f"\n{'━' * 80}")
            lines.append(f"  SECTION: {section_name}")
            lines.append(f"{'━' * 80}")
            content = self._extract_section(text, section_name)
            lines.append(content)

        lines.append(f"\n{border}")
        lines.append("  END OF 10-K SUMMARY")
        lines.append(border)
        return "\n".join(lines)

    @staticmethod
    def _error_report(ticker: str, message: str) -> str:
        border = "=" * 80
        return (
            f"{border}\n"
            f"  SEC 10-K ANALYSIS  —  {ticker}\n"
            f"{border}\n"
            f"  ERROR: {message}\n"
            f"  Check your internet connection and verify the ticker symbol.\n"
            f"{border}\n"
        )

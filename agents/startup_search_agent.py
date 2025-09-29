# agents/startup_search_agent.py
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import asyncio
from typing import List, Dict, Any
from tools.nextunicorn import nextunicorn_list, nextunicorn_company_details_batch


def run_nextunicorn_sync(limit: int = 2, headless: bool = True) -> List[Dict[str, Any]]:
    return asyncio.run(nextunicorn_list(limit=limit, headless=headless))


def run_company_details_sync(
    urls: List[str], headless: bool = True
) -> List[Dict[str, Any]]:
    return asyncio.run(nextunicorn_company_details_batch(urls, headless=headless))

from tools.nextunicorn import nextunicorn_list, nextunicorn_company_details_batch, summarize_company_text

if __name__ == "__main__":
    items = run_nextunicorn_sync(limit=2, headless=True)
    print(f"[LIST] {len(items)} items")
    for it in items:
        print("  -", it)

    urls = [it["url"] for it in items]
    details = run_company_details_sync(urls, headless=True)

    print("\n[DETAILS - STRUCTURED]")
    for d in details:
        if "error" in d:
            print(d)
        else:
            structured = summarize_company_text(d["full_text"])
            print("URL:", d["url"])
            for k, v in structured.items():
                if v:
                    print(f"## {k.upper()}\n{v}\n")
            print("----")

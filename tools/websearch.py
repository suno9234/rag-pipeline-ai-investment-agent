# tools/websearch.py
from playwright.async_api import async_playwright
from urllib.parse import quote
from tools.utils import user_agent, polite_sleep, retry_policy
from tools.fetchparse import fetch_and_parse
from tenacity import retry

SEARCH_URL = "https://www.nextunicorn.kr/finder?tab=startup&sb=70"


async def _parse_ddg_results(page, max_results: int = 10):
    # DDG DOM은 수시로 바뀜 → 보조 선택자 혼합
    items = await page.locator("a.result__a, a[data-testid='result-title-a']").all()
    results = []
    for a in items[:max_results]:
        href = await a.get_attribute("href")
        title = await a.text_content()
        # 스니펫 시도
        parent = a.locator("xpath=..").first
        snippet = ""
        try:
            sn_node = parent.locator(
                ".result__snippet, [data-testid='result-snippet']"
            ).first
            snippet = (await sn_node.text_content()) or ""
        except Exception:
            pass
        if href:
            results.append(
                {
                    "title": (title or "").strip(),
                    "url": href,
                    "snippet": (snippet or "").strip(),
                }
            )
    return results


@retry(**retry_policy)
async def web_search(
    query: str, *, max_results: int = 8, visit_top_k: int = 2, headless: bool = True
):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(user_agent=user_agent())
        page = await context.new_page()

        await page.goto(SEARCH_URL.format(q=quote(query)))
        polite_sleep()

        results = await _parse_ddg_results(page, max_results=max_results)

        await context.close()
        await browser.close()

    visited = []
    for r in results[:visit_top_k]:
        try:
            visited.append(
                await fetch_and_parse(
                    r["url"], headless=headless, sniff_network=False, max_scroll_steps=4
                )
            )
        except Exception as e:
            visited.append({"url": r["url"], "error": str(e)})

    return {"query": query, "results": results, "visited": visited}

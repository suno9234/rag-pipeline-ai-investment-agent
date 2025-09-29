# tools/fetchparse.py
from playwright.async_api import async_playwright
from tools.utils import user_agent, retry_policy
from tools.robots import is_allowed
from tools.extract import extract_basic
from tenacity import retry


async def _auto_scroll(page, max_steps=12, step_px=1200):
    for _ in range(max_steps):
        await page.evaluate(f"window.scrollBy(0, {step_px});")
        await page.wait_for_timeout(500)


@retry(**retry_policy)
async def fetch_and_parse(
    url: str,
    *,
    headless: bool = True,
    sniff_network: bool = False,
    max_scroll_steps: int = 0,
    wait_selector: str | None = None,
    timeout_ms: int = 20000,
):
    if not is_allowed(url):
        return {"url": url, "blocked_by_robots": True}

    net_logs = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(user_agent=user_agent())
        page = await context.new_page()

        if sniff_network:
            page.on(
                "response",
                lambda res: net_logs.append(
                    {
                        "url": res.url,
                        "status": res.status,
                    }
                ),
            )

        await page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
        if wait_selector:
            await page.wait_for_selector(wait_selector, timeout=timeout_ms)

        if max_scroll_steps > 0:
            await _auto_scroll(page, max_steps=max_scroll_steps)

        html = await page.content()
        data = extract_basic(html)

        await context.close()
        await browser.close()

    return {
        "url": url,
        "blocked_by_robots": False,
        "extracted": data,
        "network": net_logs,
    }

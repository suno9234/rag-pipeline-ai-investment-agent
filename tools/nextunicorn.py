# tools/nextunicorn.py
"""
NextUnicorn 전용 크롤링 툴
- 공개 API:
    * nextunicorn_list(...)                       : 스타트업 카드 리스트 수집
    * nextunicorn_company_details_batch(urls, ...) : 상세 페이지 본문 수집 배치
    * summarize_company_text(full_text)           : 상세 본문 섹션 요약
    * nextunicorn_list_and_details(... )          : (신규) 한 세션으로 리스트→디테일까지 한번에 수집
    * nextunicorn_list_and_details_sync(... )     : (신규) 동기 래퍼
- 성능 최적화:
    * 브라우저/컨텍스트 1회 재사용 (list_and_details)
    * 이미지/폰트/미디어/애널리틱스 요청 차단으로 대역폭/렌더링 시간 절감
    * storage_state 재활용으로 로그인 생략
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Iterable, TypedDict, Optional
from dotenv import load_dotenv
from playwright.async_api import async_playwright, TimeoutError as PWTimeout
from tools.utils import user_agent
import logging

load_dotenv()
log = logging.getLogger(__name__)

# ====== 공개 API 제한 ======
__all__ = [
    "nextunicorn_list",
    "nextunicorn_company_details_batch",
    "summarize_company_text",
    "nextunicorn_list_and_details",
    "nextunicorn_list_and_details_sync",
]

# ====== 선택자/상수 ======
_CARD_SEL = "a[data-event='companyCard'][data-indicator='findStartup']"
_TITLE_SEL = "span.UnicornCompanyCard_Title"
_SUMMARY_SEL = "span.UnicornCompanyCard_Summary"
_MORE_BTN = "button[data-event='finder/:tabName/seeMore'][data-tab-name='startup']"
_STORAGE_FILE = "nextunicorn_state.json"

# 로그인 페이지 요소
_EMAIL_INP_PL = "넥스트유니콘 이메일 계정을 입력해주세요."
_PASS_INP_PL = "비밀번호를 입력해주세요."
_LOGIN_BTN_TX = "로그인"

# 텍스트 정제
_TAG_TOKEN = re.compile(r"#\S+")
_ALLOWED = re.compile(r"[^0-9A-Za-z가-힣\s]")
_MSPACE = re.compile(r"\s+")

# 노이즈 제거 키워드/라벨
_NOISE_CONTAINS = [
    "무료상담신청", "IR자료 요청하기", "티켓", "티켓구매",
    "제휴 및 광고 문의", "비즈니스", "템플릿", "지원프로그램",
    "인사이트", "MyCFO", "펼쳐보기", "더 보기",
    "2025년 내 시리즈A 이상 투자유치가 목표라면",
]
_NOISE_PREFIXES = ["홈>", "홈 >", "Home>", "home>", "HOME>", "파인더>", "파인더 >"]
_TAB_LABELS = {"전체", "투자 정보", "서비스/제품", "팀 정보"}


# ====== 타입 ======
class Card(TypedDict):
    title: str
    summary: str
    url: str

class Detail(TypedDict, total=False):
    url: str
    full_text: str
    error: str

class Structured(TypedDict):
    company: str
    summary: str
    services: str
    team: str
    news: str
    funding: str
    info: str


# ====== 내부 유틸 ======
def _sanitize_text(text: Optional[str]) -> str:
    """태그/특수문자/다중 공백 제거"""
    if not text:
        return ""
    s = _TAG_TOKEN.sub(" ", text)
    s = _ALLOWED.sub(" ", s)
    s = _MSPACE.sub(" ", s).strip()
    return s

async def _ensure_logged_in(context, page):
    """
    세션 없으면 로그인 → storage_state 저장
    - .env 에 NEXTUNICORN_ID / NEXTUNICORN_PASSWORD 필요
      (비번에 # 포함 시 따옴표로 감싸기)
    """
    if Path(_STORAGE_FILE).exists():
        return

    email = os.getenv("NEXTUNICORN_ID")
    password = os.getenv("NEXTUNICORN_PASSWORD")
    if not email or not password:
        raise RuntimeError("NEXTUNICORN_ID / NEXTUNICORN_PASSWORD 필요 (.env). 비번에 #가 있으면 따옴표로 감싸세요.")

    await page.goto("https://www.nextunicorn.kr/login", wait_until="domcontentloaded", timeout=45000)

    form = page.locator("form:has(input[name='email']):has(input[name='password'])").first
    await form.wait_for(state="visible", timeout=20000)

    email_inp = form.get_by_placeholder(_EMAIL_INP_PL)
    pass_inp = form.get_by_placeholder(_PASS_INP_PL)

    await email_inp.fill(email)
    await pass_inp.fill(password)

    login_btn = form.get_by_role("button", name=_LOGIN_BTN_TX, exact=True)
    clicked = False
    try:
        await login_btn.wait_for(state="attached", timeout=10000)
        for _ in range(40):
            if await login_btn.is_enabled():
                await login_btn.click(timeout=5000)
                clicked = True
                break
            await page.wait_for_timeout(100)
    except PWTimeout:
        clicked = False

    if not clicked:
        await pass_inp.press("Enter")

    await page.wait_for_load_state("networkidle", timeout=20000)
    try:
        await context.storage_state(path=_STORAGE_FILE)
    except Exception as e:
        log.debug("storage_state save failed: %s", e)


async def _extract_card(card) -> Card:
    """카드 엘리먼트에서 title/summary/url 추출"""
    href = await card.get_attribute("href")
    full = href if (href or "").startswith("http") else f"https://www.nextunicorn.kr{href}"
    title = ""
    summary = ""
    try:
        t = await card.locator(_TITLE_SEL).first.text_content()
        if t:
            title = _sanitize_text(t)
    except Exception as e:
        log.debug("title parse fail: %s", e)
    try:
        s = await card.locator(_SUMMARY_SEL).first.text_content()
        if s:
            summary = _sanitize_text(s)
    except Exception as e:
        log.debug("summary parse fail: %s", e)
    return {"title": title, "summary": summary, "url": full}


def _ensure_all_tab(url: str) -> str:
    """상세 URL에 ?tab=all 파라미터 강제 부여"""
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
    pr = urlparse(url)
    qs = parse_qs(pr.query)
    qs["tab"] = ["all"]
    new_q = urlencode({k: v[0] if isinstance(v, list) else v for k, v in qs.items()})
    return urlunparse((pr.scheme, pr.netloc, pr.path, pr.params, new_q, pr.fragment))


async def _grab_full_text(page) -> str:
    """상세 페이지에서 본문 텍스트를 최대한 펼쳐 수집"""
    await page.wait_for_load_state("domcontentloaded")
    await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
    await page.wait_for_timeout(700)

    # 헤더/푸터/사이드 제거 + 푸터 텍스트 패턴 제거
    await page.evaluate("""
    (() => {
      document.querySelectorAll('header, footer, nav, aside').forEach(el => el.remove());
      const killByText = (needle) => {
        const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT);
        let nodes = [];
        while (walker.nextNode()) nodes.push(walker.currentNode);
        for (const el of nodes) {
          const t = (el.innerText || '').trim();
          if (t && t.includes(needle)) {
            (el.closest('footer, section, div') || el).remove();
            break;
          }
        }
      };
      killByText("주식회사 넥스트유니콘");
    })();
    """)

    # '펼쳐보기'류 버튼 펼치기
    for btn_text in ["펼쳐보기", "더 보기", "투자 정보 더 보기", "서비스/제품 정보 더 보기", "팀 정보 더 보기"]:
        loc = page.get_by_role("button", name=btn_text)
        try:
            if await loc.count() > 0:
                for i in range(await loc.count()):
                    b = loc.nth(i)
                    if await b.is_visible():
                        try:
                            await b.click(timeout=1500)
                            await page.wait_for_timeout(300)
                        except Exception:
                            pass
        except Exception:
            pass

    await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
    await page.wait_for_timeout(500)

    try:
        return await page.locator("body").inner_text()
    except Exception:
        return await page.inner_text("html")


def _strip_noise_lines(text: str) -> str:
    """불필요 라인/중복/빈줄 정리"""
    t = text.replace("\u200b", "").replace("\xa0", " ")
    lines = [l.rstrip() for l in t.splitlines()]
    keep, prev = [], None
    for l in lines:
        ls = l.strip()
        if not ls:
            keep.append(ls); prev = ls; continue
        if ls in _TAB_LABELS:                      # 탭 라벨 제거
            continue
        if any(ls.startswith(p) for p in _NOISE_PREFIXES):
            continue
        if any(k in ls for k in _NOISE_CONTAINS):
            continue
        if ls.startswith("#") and len(ls) < 50:
            continue
        if ls == prev:
            continue
        keep.append(ls); prev = ls

    out, blank = [], False
    for l in keep:
        if not l:
            if blank:
                continue
            blank = True
        else:
            blank = False
        out.append(l)
    return "\n".join(out).strip()


def _clean_text(raw: str) -> str:
    """사소한 계정명 흔적/노이즈 제거"""
    if not raw:
        return ""
    raw = raw.replace("신순호", "")  # 로그인명 흔적 제거 가능성
    return _strip_noise_lines(raw)


def _section(text: str, start_kw: str, end_kws: list[str]) -> str:
    """'소개'→'투자 정보' 같은 섹션 경계로 부분 문자열 추출"""
    if start_kw not in text:
        return ""
    seg = text.split(start_kw, 1)[-1]
    for end in end_kws:
        if end in seg:
            seg = seg.split(end, 1)[0]
            break
    return seg.strip()


def _compact(s: str) -> str:
    """3줄 이상 연속 개행 → 2줄로 압축"""
    return re.sub(r"\n{3,}", "\n\n", (s or "").strip())


def _dedupe_lines(s: str) -> str:
    """중복 라인 제거"""
    seen, out = set(), []
    for l in (s.splitlines() if s else []):
        if l not in seen:
            out.append(l); seen.add(l)
    return "\n".join(out).strip()


# ====== 공개 API (기존) ======
async def nextunicorn_list(
    url: str = "https://www.nextunicorn.kr/finder?tab=startup&sb=70",
    *,
    headless: bool = True,
    limit: int = 50,
) -> List[Card]:
    """스타트업 카드 리스트 수집"""
    results: List[Card] = []
    seen = set()
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(
            user_agent=user_agent(),
            storage_state=_STORAGE_FILE if Path(_STORAGE_FILE).exists() else None,
        )
        page = await context.new_page()
        await _ensure_logged_in(context, page)

        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        await page.wait_for_selector(_CARD_SEL, timeout=45000)
        await page.wait_for_timeout(500)

        async def collect():
            nonlocal results, seen
            cards = await page.locator(_CARD_SEL).all()
            for c in cards:
                href = await c.get_attribute("href")
                if not href or href in seen:
                    continue
                seen.add(href)
                results.append(await _extract_card(c))
                if len(results) >= limit:
                    break

        await collect()
        while len(results) < limit:
            btn = page.locator(_MORE_BTN)
            if await btn.count() == 0 or not await btn.first.is_visible():
                break
            try:
                await btn.first.click(timeout=4000)
            except Exception:
                await page.evaluate("window.scrollBy(0, 1200)")
            await page.wait_for_timeout(1200)
            await collect()

        await context.close()
        await browser.close()
    return results[:limit]


async def nextunicorn_company_details_batch(
    urls: Iterable[str],
    *,
    headless: bool = True,
) -> List[Detail]:
    """상세 페이지 본문 수집 배치"""
    out: List[Detail] = []
    urls = [_ensure_all_tab(u) for u in urls]
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(
            user_agent=user_agent(),
            storage_state=_STORAGE_FILE if Path(_STORAGE_FILE).exists() else None,
        )
        page = await context.new_page()
        await _ensure_logged_in(context, page)

        for u in urls:
            try:
                await page.goto(u, wait_until="domcontentloaded", timeout=45000)
                text = await _grab_full_text(page)
                out.append({"url": u, "full_text": text})
            except Exception as e:
                out.append({"url": u, "error": str(e)})

        await context.close()
        await browser.close()
    return out


def summarize_company_text(full_text: str) -> Structured:
    """상세 본문에서 섹션별 요약 텍스트 추출"""
    t = _clean_text(full_text)
    head = "\n".join([l for l in t.splitlines()[:10] if l.strip()])

    summary   = _section(t, "소개", ["투자 정보", "서비스/제품 정보", "팀 정보", "기업 소식", "회사 정보"])
    funding   = _section(t, "투자 정보", ["서비스/제품 정보", "팀 정보", "기업 소식", "회사 정보"])
    services  = _section(t, "서비스/제품 정보", ["팀 정보", "기업 소식", "회사 정보"])
    team      = _section(t, "팀 정보", ["기업 소식", "회사 정보"])
    news      = _section(t, "기업 소식", ["회사 정보"])
    info      = _section(t, "회사 정보", [])

    def polish(s: str) -> str:
        return _dedupe_lines(_compact(s))

    return {
        "company": polish(head),
        "summary": polish(summary),
        "services": polish(services),
        "team": polish(team),
        "news": polish(news),
        "funding": polish(funding),
        "info": polish(info),
    }


# ====== 공개 API (신규): 한 세션으로 리스트+디테일 ======
async def nextunicorn_list_and_details(
    *,
    limit: int = 2,
    headless: bool = True,
) -> Dict[str, Any]:
    """
    한 번의 브라우저/컨텍스트로
    1) 리스트 수집 → 2) URL 추출 → 3) 상세 본문 수집까지 수행.
    - 리소스 차단(이미지/폰트/미디어/애널리틱스)로 속도 개선.
    - 반환: {"items":[Card...], "details":[Detail...]}
    """
    items: List[Card] = []
    details: List[Detail] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(
            user_agent=user_agent(),
            storage_state=_STORAGE_FILE if Path(_STORAGE_FILE).exists() else None,
        )

        # 리소스 차단으로 네트워크/렌더링 비용 절감
        async def _route(route):
            req = route.request
            url = req.url
            rtype = req.resource_type
            if rtype in {"image", "media", "font"} or any(
                k in url for k in [
                    "google-analytics", "gtag", "hotjar", "segment",
                    "facebook", "doubleclick", "googletagmanager"
                ]
            ):
                return await route.abort()
            return await route.continue_()

        await context.route("**/*", _route)

        page = await context.new_page()
        await _ensure_logged_in(context, page)

        # 1) 리스트
        await page.goto("https://www.nextunicorn.kr/finder?tab=startup&sb=70", wait_until="domcontentloaded", timeout=45000)
        await page.wait_for_selector(_CARD_SEL, timeout=45000)
        await page.wait_for_timeout(400)

        seen = set()
        async def collect():
            nonlocal items, seen
            cards = await page.locator(_CARD_SEL).all()
            for c in cards:
                href = await c.get_attribute("href")
                if not href or href in seen:
                    continue
                seen.add(href)
                items.append(await _extract_card(c))
                if len(items) >= limit:
                    break

        await collect()
        while len(items) < limit:
            btn = page.locator(_MORE_BTN)
            if await btn.count() == 0 or not await btn.first.is_visible():
                break
            try:
                await btn.first.click(timeout=3500)
            except Exception:
                await page.evaluate("window.scrollBy(0, 1200)")
            await page.wait_for_timeout(900)
            await collect()

        # 2) 디테일 (같은 페이지/세션 재사용)
        urls = [_ensure_all_tab(it["url"]) for it in items if it.get("url")]
        for u in urls:
            try:
                await page.goto(u, wait_until="domcontentloaded", timeout=45000)
                txt = await _grab_full_text(page)
                details.append({"url": u, "full_text": txt})
            except Exception as e:
                details.append({"url": u, "error": str(e)})

        await context.close()
        await browser.close()

    return {"items": items, "details": details}


def nextunicorn_list_and_details_sync(
    *,
    limit: int = 2,
    headless: bool = True,
) -> Dict[str, Any]:
    """동기 환경에서 바로 호출 가능한 래퍼"""
    import asyncio
    return asyncio.run(nextunicorn_list_and_details(limit=limit, headless=headless))

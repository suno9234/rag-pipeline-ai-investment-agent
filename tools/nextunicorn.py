# tools/nextunicorn.py
import os, re
from pathlib import Path
from typing import List, Dict, Any, Iterable
from dotenv import load_dotenv
from playwright.async_api import async_playwright, TimeoutError as PWTimeout
from tools.utils import user_agent

load_dotenv()

# 리스트 페이지(카드 수집)용 선택자
CARD_SEL = "a[data-event='companyCard'][data-indicator='findStartup']"
TITLE_SEL = "span.UnicornCompanyCard_Title"
SUMMARY_SEL = "span.UnicornCompanyCard_Summary"
MORE_BTN = "button[data-event='finder/:tabName/seeMore'][data-tab-name='startup']"

# 로그인 페이지 요소
EMAIL_INP_PL = "넥스트유니콘 이메일 계정을 입력해주세요."
PASS_INP_PL = "비밀번호를 입력해주세요."
LOGIN_BTN_TX = "로그인"
STORAGE_FILE = "nextunicorn_state.json"

# 텍스트 정제: #태그 제거 + 숫자/영문/한글/공백만 남김
_TAG_TOKEN = re.compile(r"#\S+")
_ALLOWED = re.compile(r"[^0-9A-Za-z가-힣\s]")
_MSPACE = re.compile(r"\s+")


def sanitize_text(text: str | None) -> str:
    if not text:
        return ""
    s = _TAG_TOKEN.sub(" ", text)
    s = _ALLOWED.sub(" ", s)
    s = _MSPACE.sub(" ", s).strip()
    return s


# ------------------ 로그인 ------------------
async def _ensure_logged_in(context, page):
    """세션 없으면 로그인 → storage_state 저장."""
    if Path(STORAGE_FILE).exists():
        return

    email = os.getenv("NEXTUNICORN_ID")
    password = os.getenv("NEXTUNICORN_PASSWORD")
    if not email or not password:
        raise RuntimeError(
            "NEXTUNICORN_ID / NEXTUNICORN_PASSWORD 필요 (.env). 비번에 #가 있으면 따옴표로 감싸세요."
        )

    await page.goto(
        "https://www.nextunicorn.kr/login", wait_until="domcontentloaded", timeout=45000
    )

    # 로그인 폼 스코프
    form = page.locator(
        "form:has(input[name='email']):has(input[name='password'])"
    ).first
    await form.wait_for(state="visible", timeout=20000)

    email_inp = form.get_by_placeholder(EMAIL_INP_PL)
    pass_inp = form.get_by_placeholder(PASS_INP_PL)

    await email_inp.fill("")
    await email_inp.fill(email)
    await pass_inp.fill("")
    await pass_inp.fill(password)

    login_btn = form.get_by_role("button", name=LOGIN_BTN_TX, exact=True)
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
        await context.storage_state(path=STORAGE_FILE)
    except Exception:
        pass


# ------------------ 리스트 수집 ------------------
async def _extract_card(card) -> Dict:
    href = await card.get_attribute("href")
    full = (
        href if (href or "").startswith("http") else f"https://www.nextunicorn.kr{href}"
    )
    title = ""
    summary = ""
    try:
        t = await card.locator(TITLE_SEL).first.text_content()
        if t:
            title = sanitize_text(t)
    except:
        pass
    try:
        s = await card.locator(SUMMARY_SEL).first.text_content()
        if s:
            summary = sanitize_text(s)
    except:
        pass
    return {"title": title, "summary": summary, "url": full}


async def nextunicorn_list(
    url: str = "https://www.nextunicorn.kr/finder?tab=startup&sb=70",
    *,
    headless: bool = True,
    limit: int = 50,
) -> List[Dict]:
    """로그인 후 '더 보기' 반복 클릭하여 카드 최대 limit개 수집 (storage_state 적용)."""
    results: List[Dict] = []
    seen = set()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(
            user_agent=user_agent(),
            storage_state=STORAGE_FILE if Path(STORAGE_FILE).exists() else None,
        )
        page = await context.new_page()

        await _ensure_logged_in(context, page)

        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        await page.wait_for_selector(CARD_SEL, timeout=45000)
        await page.wait_for_timeout(500)

        async def collect():
            nonlocal results, seen
            cards = await page.locator(CARD_SEL).all()
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
            btn = page.locator(MORE_BTN)
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


# ------------------ 상세: 전체 텍스트 통 긁기 ------------------
def _ensure_all_tab(url: str) -> str:
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

    pr = urlparse(url)
    qs = parse_qs(pr.query)
    qs["tab"] = ["all"]
    new_q = urlencode({k: v[0] if isinstance(v, list) else v for k, v in qs.items()})
    return urlunparse((pr.scheme, pr.netloc, pr.path, pr.params, new_q, pr.fragment))


async def _grab_full_text(page) -> str:
    """헤더/푸터/사이드 제거 + '펼쳐보기/더 보기' 확장 후 본문 텍스트 수집."""
    # 0) 전체 로딩 대기 & 1차 스크롤
    await page.wait_for_load_state("domcontentloaded")
    await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
    await page.wait_for_timeout(700)

    # 1) 헤더/푸터/사이드 제거 (사이트 공통 노이즈)
    await page.evaluate("""
    (() => {
      // 표준 태그
      document.querySelectorAll('header, footer, nav, aside').forEach(el => el.remove());
      // NextUnicorn 공용 푸터 텍스트 포함 컨테이너 제거
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

    # 2) 섹션 콘텐츠 확장: '펼쳐보기' / '... 더 보기' 클릭
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

    # 3) 해시태그 칩/필터 같은 노이즈 요소 제거(시각 칩, 토큰)
    await page.evaluate("""
    (() => {
      const looksLikeTag = (txt) => txt && txt.trim().startsWith("#");
      document.querySelectorAll("a, span, div").forEach(el => {
        const t = (el.innerText || "").trim();
        if (t && looksLikeTag(t) && t.length < 50) el.remove();
      });
    })();
    """)

    # 4) 마지막으로 한번 더 스크롤(지연 로드 대비)
    await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
    await page.wait_for_timeout(500)

    # 5) 본문 텍스트 수집 (정제는 요약 단계에서)
    try:
        full = await page.locator("body").inner_text()
    except Exception:
        full = await page.inner_text("html")
    return full



async def nextunicorn_company_details_batch(
    urls: Iterable[str],
    *,
    headless: bool = True,
) -> List[Dict[str, Any]]:
    """여러 회사 상세 페이지에서 '전체 텍스트'만 통으로 수집."""
    out: List[Dict[str, Any]] = []
    urls = [_ensure_all_tab(u) for u in urls]

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(
            user_agent=user_agent(),
            storage_state=STORAGE_FILE if Path(STORAGE_FILE).exists() else None,
        )
        page = await context.new_page()

        await _ensure_logged_in(context, page)

        for u in urls:
            try:
                await page.goto(u, wait_until="domcontentloaded", timeout=45000)
                # 핵심 텍스트 수집
                text = await _grab_full_text(page)
                out.append({"url": u, "full_text": text})
            except Exception as e:
                out.append({"url": u, "error": str(e)})

        await context.close()
        await browser.close()

    return out


import re

# 상단 배너/CTA/브레드크럼/탭 라벨 등 노이즈 키워드
_NOISE_CONTAINS = [
    "무료상담신청",
    "IR자료 요청하기",
    "티켓", "티켓구매",
    "제휴 및 광고 문의",
    "비즈니스", "템플릿", "지원프로그램", "인사이트", "MyCFO",
    "펼쳐보기", "더 보기",
    # 페이지 상단 배너 문구들
    "2025년 내 시리즈A 이상 투자유치가 목표라면",
]

# 브레드크럼/네비 prefix
_NOISE_PREFIXES = ["홈>", "홈 >", "Home>", "home>", "HOME>", "파인더>", "파인더 >"]

# 탭 라벨(라인 전체가 이 값이면 제거)
_TAB_LABELS = {"전체", "투자 정보", "서비스/제품", "팀 정보"}

def _strip_noise_lines(text: str) -> str:
    """라인 단위로 노이즈 제거 + 중복/공백 정리."""
    # 유니코드 공백/가로줄 정리
    t = text.replace("\u200b", "").replace("\xa0", " ")
    lines = [l.rstrip() for l in t.splitlines()]
    keep = []
    prev = None

    for l in lines:
        ls = l.strip()

        # 빈 줄은 일단 유지 (나중에 압축)
        if not ls:
            keep.append(ls); prev = ls; continue

        # 탭 라벨(정확히 일치) 제거
        if ls in _TAB_LABELS:
            continue

        # 브레드크럼/네비 prefix 제거
        if any(ls.startswith(p) for p in _NOISE_PREFIXES):
            continue

        # 포함 키워드 노이즈 제거
        if any(k in ls for k in _NOISE_CONTAINS):
            continue

        # 해시칩/짧은 토큰류 제거
        if ls.startswith("#") and len(ls) < 50:
            continue

        # 바로 이전 라인과 완전 동일하면 제거
        if ls == prev:
            continue

        keep.append(ls)
        prev = ls

    # 연속 빈줄 1줄로 축소
    out = []
    blank = False
    for l in keep:
        if not l:
            if blank:  # 이미 빈 줄이면 스킵
                continue
            blank = True
        else:
            blank = False
        out.append(l)
    return "\n".join(out).strip()

def clean_text(raw: str) -> str:
    """간단 클린업: 로그인명/노이즈 라인 정리 (푸터/헤더 제거는 크롤 단계에서 처리됨)."""
    if not raw: 
        return ""
    # 혹시 남아있을 수 있는 로그인명 제거
    raw = raw.replace("신순호", "")
    return _strip_noise_lines(raw)

def _section(text: str, start_kw: str, end_kws: list[str]) -> str:
    """start_kw 이후부터 end_kws 중 첫 등장 전까지 슬라이스."""
    if start_kw not in text:
        return ""
    seg = text.split(start_kw, 1)[-1]
    for end in end_kws:
        if end in seg:
            seg = seg.split(end, 1)[0]
            break
    return seg.strip()

def _compact(s: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", (s or "").strip())

def _dedupe_lines(s: str) -> str:
    seen, out = set(), []
    for l in (s.splitlines() if s else []):
        if l not in seen:
            out.append(l); seen.add(l)
    return "\n".join(out).strip()

def summarize_company_text(full_text: str) -> dict:
    t = clean_text(full_text)

    # 상단 10줄 정도에서 회사명/요약 후보만 간략히
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

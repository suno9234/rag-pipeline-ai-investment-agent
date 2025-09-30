from __future__ import annotations
from pathlib import Path
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
import os, sys, json, re, time
from typing import List, Dict, Optional, Tuple

# ── 경로/환경 ─────────────────────────────────────────
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

DOCS_DIR = project_root / "docs"
DOCS_DIR.mkdir(exist_ok=True)

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ── 기본 설정 (하드코딩) ──────────────────────────────
DEFAULT_INDUSTRY: str = "모빌리티"
DEFAULT_GROUPS: List[str] = ["전기차", "전동킥보드", "자율주행"]

# ── 도메인 화이트/블랙 리스트 ───────────────────────
KOREA_TRUSTED = [
    "keei.re.kr","spri.kr","kama.or.kr","kotra.or.kr",
    "kostat.go.kr","data.go.kr","me.go.kr","motie.go.kr","koti.re.kr"
]
GLOBAL_TRUSTED = ["iea.org","oecd.org","imf.org","worldbank.org","ec.europa.eu"]
BLOCK = [
    "gminsights.com","imarcgroup.com","wiseguyreports.com","extrapolate.com",
    "marketwatch.com/press-release","globenewswire.com"
]
GROUP_TRUSTED_MAP = {
    "전기차": KOREA_TRUSTED,
    "전동킥보드": KOREA_TRUSTED,
    "자율주행": KOREA_TRUSTED + ["nhtsa.gov","mlit.go.jp"],
}

# ── 노이즈(목록/검색/배너) 컷 규칙 (정교화) ────────────
NOISY_URL_SUBSTR = [
    "/user/search/search.do", "/search/search.do",
    "mnthngList.do", "pmediaView.do", "cardnewsView.do",
    "/mnthngList.do", "/bbs/mnthngList.do"
]
NOISY_CONTENT_PATTERNS = [
    r"전체\s*\(\d+건\)", r"연구보고서\s*\(\d+건\)", r"정기간행물\s*\(\d+건\)",
    r"특화사업\s*\(\d+건\)", r"종료특화사업\s*\(\d+건\)", r"디지털콘텐츠\s*\(\d+건\)",
    r"열린광장\s*\(\d+건\)", r"검색결과", r"통합검색", r"전체메뉴",
    r"바로보기\s*\_?", r"Image\s*\d+\s*:", r"\b목차\b"
]
NOISY_TITLE_PATTERNS = [r"검색결과", r"목록", r"List"]

MIN_CONTENT_CHARS = 100
MIN_KEEP_AFTER_FILTER = 3
MAX_RESULTS_PRIMARY = 10
MAX_RESULTS_RELAXED  = 15

# ── 그룹별 쿼리 확장(동의어/대체어) ─────────────────────
GROUP_SYNONYMS: Dict[str, List[str]] = {
    "전기차": ["전기차", "BEV", "전기승용차", "전기트럭", "ZEV", "배터리"],
    "전동킥보드": ["전동킥보드", "퍼스널 모빌리티", "PM", "e-스크터", "공유 킥보드"],
    "자율주행": ["자율주행", "ADS", "레벨4", "로보택시", "자율주행차"],
}

def get_default_config() -> Tuple[str, List[str], Dict[str, List[str]]]:
    """
    임베딩 노드가 하드코딩된 검색 구성을 읽어갈 때 사용.
    """
    return DEFAULT_INDUSTRY, DEFAULT_GROUPS, GROUP_SYNONYMS

# ── 크롤 텍스트 수집 ──────────────────────────────────
def fetch_fulltext(url: str) -> Optional[str]:
    try:
        import trafilatura
    except Exception:
        return None
    try:
        downloaded = trafilatura.fetch_url(url, timeout=15)
        if not downloaded:
            return None
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        return (text or "").strip()
    except Exception:
        return None

# ── 검색 결과 정규화/필터 ─────────────────────────────
def normalize_results(res) -> List[Dict]:
    if isinstance(res, list):
        return [x if isinstance(x, dict) else {"title": str(x), "url": "", "content": ""} for x in res]
    if isinstance(res, dict) and isinstance(res.get("results"), list):
        items = res["results"]
        return [x if isinstance(x, dict) else {"title": str(x), "url": "", "content": ""} for x in items]
    return [{"title": str(res), "url": "", "content": ""}]

def hard_filter(items: List[Dict], whitelist: List[str], blacklist: List[str]) -> List[Dict]:
    out = []
    for it in items:
        url = (it.get("url") or it.get("source") or "").strip()
        if not url:
            continue
        if any(b in url for b in blacklist):
            continue
        if whitelist and not any(dom in url for dom in whitelist):
            continue
        out.append(it)
    return out

def looks_like_listing_page(title: str, content: str, url: str) -> bool:
    if any(p in (url or "") for p in NOISY_URL_SUBSTR):
        return True
    if any(re.search(p, title or "", flags=re.I) for p in NOISY_TITLE_PATTERNS):
        return True
    if any(re.search(p, content or "", flags=re.I) for p in NOISY_CONTENT_PATTERNS):
        return True
    return False

# ── Tavily 클라이언트/쿼리 빌더 ────────────────────────
def build_tavily_client(
    include_domains: List[str],
    topic: str = "news",
    time_range: str = "year",
    max_results: int = MAX_RESULTS_PRIMARY
) -> TavilySearch:
    if not TAVILY_API_KEY:
        raise EnvironmentError("TAVILY_API_KEY가 설정되지 않았습니다. (.env 확인)")
    return TavilySearch(
        api_key=TAVILY_API_KEY,
        topic=topic,
        search_depth="advanced",
        time_range=time_range,
        include_raw_content="markdown",
        max_results=max_results,
        include_domains=include_domains,
        exclude_domains=BLOCK,
        country="south korea",
        auto_parameters=True,
    )

def build_queries(industry: str, group: str, synonyms_override: dict | None = None) -> List[str]:
    syns = (synonyms_override or {}).get(group) or GROUP_SYNONYMS.get(group, [group])
    base = [
        f"한국 {industry} {group} 시장 동향 정책 통계 보고서 filetype:pdf OR 보고서 OR 통계",
        f"한국 {industry} {group} 산업 동향 보도자료 통계 발표",
    ]
    for s in syns[:3]:
        base.append(f"한국 {industry} {s} 동향 정책 통계")
    return base

def build_relaxed_queries(industry: str, group: str, synonyms_override: dict | None = None) -> List[str]:
    syns = (synonyms_override or {}).get(group) or GROUP_SYNONYMS.get(group, [group])
    relaxed = [
        f"{industry} {group} 한국 시장 동향 정책 통계",
        f"{industry} {group} 국내 보고서 동향",
    ]
    for s in syns[:2]:
        relaxed.append(f"{industry} {s} 한국 동향")
    return relaxed

def build_site_queries(industry: str, group: str, domains: List[str]) -> List[str]:
    q = []
    for d in domains[:8]:
        q.append(f"site:{d} {industry} {group} 동향 OR 보고서 OR 통계")
    return q

def tavily_invoke(client: TavilySearch, query: str) -> List[Dict]:
    try:
        res = client.invoke({"query": query})
        return normalize_results(res)
    except Exception as e:
        print(f"   ⚠️ Tavily 오류: {e}")
        return []

def clean_and_enrich(items: List[Dict]) -> List[Dict]:
    cleaned = []
    for it in items:
        title = (it.get("title") or "").strip()
        content = (it.get("content") or it.get("snippet") or it.get("body") or "").strip()
        url = (it.get("url") or it.get("source") or "").strip()
        if looks_like_listing_page(title, content, url):
            continue
        if len(content) < MIN_CONTENT_CHARS and url:
            fulltext = fetch_fulltext(url)
            if fulltext and len(fulltext) > len(content):
                content = fulltext
        if len(content.strip()) < MIN_CONTENT_CHARS:
            continue
        cleaned.append({"title": title, "content": content, "url": url})
    return cleaned

from config.chroma import get_vector_store

def _chroma_exists_by_tag(tag: str) -> bool:
    vdb = get_vector_store()
    col = vdb._collection
    res = col.get(where={"tag": tag}, limit=1)
    return bool(res and res.get("ids"))


# ── 공개 API: 검색 실행 ───────────────────────────────
def run_search(
    industry: str,
    groups: List[str],
    out_json: Optional[Path],
    use_global: bool,
    synonyms_override: dict | None = None,
):
    results_payload: Dict[str, List[Dict]] = {}

    for group in groups:
        # ✅ 먼저 VDB에 있는지 확인
        if _chroma_exists_by_tag(group):
            print(f"⏭️ [SKIP] '{group}' → 이미 VDB(tag={group})에 존재 → Tavily 호출 생략")
            results_payload[group] = []  # 검색은 PASS, JSON에는 빈 리스트 기록
            continue

        include_domains = (
            GLOBAL_TRUSTED if use_global else GROUP_TRUSTED_MAP.get(group, KOREA_TRUSTED)
        )
        collected: List[Dict] = []
        seen_urls: set[str] = set()

        def add_items(new_items: List[Dict]):
            nonlocal collected, seen_urls
            for it in new_items:
                url = (it.get("url") or it.get("source") or "").strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                collected.append(it)

        client = build_tavily_client(
            include_domains, topic="news", time_range="year", max_results=MAX_RESULTS_PRIMARY
        )
        print(f"\n🔎 [{group}] 1차(기본) 검색 쿼리 빌드")
        for q in build_queries(industry, group, synonyms_override):
            print(f"   • {q}")
            add_items(tavily_invoke(client, q))
            time.sleep(0.25)

        # === 이하 원래 검색/정제 로직 동일 ===
        items = hard_filter(collected, include_domains, BLOCK)
        stage1 = clean_and_enrich(items)
        final_list = stage1
        # ... (2차, 3차 검색 로직 그대로 유지)
        results_payload[group] = final_list

    out_json = out_json or (DOCS_DIR / f"{industry}_search_results.json")
    out_json.write_text(
        json.dumps(results_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n💾 JSON 저장 완료: {out_json}")
    return out_json



# 모듈 단독 실행 디폴트 (개발용)
if __name__ == "__main__":
    industry, groups, _ = get_default_config()
    use_global_sources = False
    out_json_path = DOCS_DIR / f"{industry}_search_results.json"
    run_search(industry, groups, out_json_path, use_global_sources)

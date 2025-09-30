# agents/startup_search_agent.py
from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional, Tuple
import json
import re
import os
import traceback
from state import State

# ── 1) 크롤링 툴 ────────────────────────────────────────────────────────────────
from tools.nextunicorn import (
    nextunicorn_list,                    # ✅ 리스트만
    nextunicorn_company_details_batch,   # ✅ 필요 URL만 상세
)

# ── 2) 벡터 스토어 / 저장 레이어 ─────────────────────────────────────────────
from config.chroma import get_vector_store
from repositories.chroma_repo import (
    find_company_exact_or_similar,
    upsert_company_profile,  # repo 내부에서 tags(list)->" | " 문자열로 변환되어야 함
)

# ── 3) LLM (섹션 정리 + 태그 동시 생성) ───────────────────────────────────────
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 환경변수 OPENAI_API_KEY 필요
_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# 한 번에 처리: cleaned 섹션 + tags 생성
_COMBINED_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "너는 벤처캐피탈 애널리스트다. 입력으로 제공되는 '회사 원문 텍스트(raw_text)'만을 기반으로 "
            "아래 두 출력을 동시에 생성하라. 섹션 제목이 없어도 원문 전체에서 의미 단위로 추출하라.\n\n"
            "[출력 스키마]\n"
            "{{\n"
            '  "cleaned": {{\n'
            '    "summary": "...",\n'
            '    "services": "...",\n'
            '    "team": "...",\n'
            '    "funding": "...",\n'
            '    "news": "...",\n'
            '    "info": "...",\n'
            '    "company": "..."\n'
            '    "tags": "태그1 | 태그2 | 태그3"\n'
            "  }},\n"
            "}}\n\n"
            "[정리 규칙]\n"
            "1) cleaned 8키는 항상 포함.\n"
            "2) 해시태그/URL/광고·이벤트/내비게이션 제거, 중복 제거, 3+개행→1, 간단 맞춤법 보정.\n"
            "3) 각 섹션 최대 800자, 과장은 줄이고 사실 위주. 근거 없으면 빈 문자열(\"\").\n"
            "4) funding은 문서 어디에 있어도(소개/뉴스 등) 시리즈/라운드/누적 투자/투자 금액/투자자 신호를 모아 요약.\n\n"
            "[태그 규칙]\n"
            "tags는 한국어 2~4어절, 최대 3개. ' | ' 구분자로 연결.\n\n"
            "[출력 형식]\n"
            "위 JSON만 출력(코드펜스 금지).",
        ),
        (
            "user",
            "회사명: {name}\n힌트(선택): {hint}\n\n원문 텍스트(raw_text):\n{raw_text}",
        ),
    ]
)

# ── 4) 상태 타입 ──────────────────────────────────────────────────────────────
class State(TypedDict, total=False):
    input_text: str
    limit: int
    headless: bool
    emit_raw: bool
    items: List[Dict[str, Any]]
    details: List[Dict[str, Any]]
    errors: List[str]

# ── 5) 내부 유틸 ──────────────────────────────────────────────────────────────
def _parse_limit_from_text(text: Optional[str], default: int = 2) -> int:
    if not text:
        return max(1, default)
    m = re.search(r"(\d+)\s*개", text)
    if not m:
        return max(1, default)
    v = int(m.group(1))
    return max(1, v)

def _normalize_all_tab(url: Optional[str]) -> str:
    if not url:
        return ""
    if "?tab=all" in url:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}tab=all"

def _local_tidy(s: str) -> str:
    """LLM 실패 대비 로컬 정리기(최소 방어)."""
    if not s:
        return ""
    s = re.sub(r"https?://\S+", " ", s)  # URL 제거
    s = re.sub(r"#\S+", " ", s)          # 해시태그 제거
    s = re.sub(r"\s{2,}", " ", s)        # 다중 공백 1개로
    s = re.sub(r"\n{3,}", "\n", s)       # 3개 이상 개행 1개로
    return s.strip()[:1000]

def _log(emit: bool, *args):
    if emit:
        print(*args)

# ── 6) LLM: 정리+태깅 통합 호출 ──────────────────────────────────────────────
_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)

def _clean_and_tag_via_llm(
    name: str,
    raw_text: str,
    hint: str = "",
    *,
    emit: bool = False,
) -> Tuple[Dict[str, str], List[str]]:
    import json as _json

    _log(emit, "[llm] combined: invoke start; raw_text_len:", len(raw_text))
    try:
        msgs = _COMBINED_PROMPT.format_messages(
            name=name,
            hint=hint or "",
            raw_text=raw_text,
        )
        out = _llm.invoke(msgs).content or ""
        _log(emit, "[llm] combined: raw output:", (out[:240].replace("\n", " ") + " ..."))

        # 코드펜스에 감싸오는 경우 대비
        m = _JSON_FENCE.search(out)
        if m:
            out = m.group(1)

        data = _json.loads(out) if out else {}
        cleaned_in = (data.get("cleaned") or {}) if isinstance(data, dict) else {}
        tags_in = (data.get("tags") or []) if isinstance(data, dict) else []

        cleaned: Dict[str, str] = {}
        for k in ["summary", "services", "team", "funding", "news", "info", "company"]:
            v = str((cleaned_in.get(k) or "")).strip()
            v = _local_tidy(v)
            if len(v) > 800:
                v = v[:800]
            cleaned[k] = v

        tags: List[str] = []
        if isinstance(tags_in, list):
            for t in tags_in[:3]:
                if isinstance(t, str):
                    tt = t.strip()
                    if tt:
                        tags.append(tt)

        _log(emit, "[llm] combined: done; lens:",
             {k: len(cleaned.get(k, "")) for k in cleaned},
             "tags:", tags)
        return cleaned, tags

    except Exception as e:
        _log(emit, "[llm] combined: ERROR:", e)
        traceback.print_exc()
        return (
            {k: "" for k in ["summary", "services", "team", "funding", "news", "info", "company"]},
            [],
        )

# ── 7) 메인 에이전트 ──────────────────────────────────────────────────────────
def startup_search_agent(state: State) -> State:
    # 1) 입력파싱
    limit: int = state.get("limit") or _parse_limit_from_text(
        state.get("input_text"), default=2
    )
    headless: bool = state.get("headless", True)
    emit_raw: bool = state.get("emit_raw", False)

    _log(emit_raw, "[boot] input_text=", state.get("input_text"))
    _log(emit_raw, "[boot] limit=", limit, "headless=", headless)
    _log(emit_raw, "[env]  VDB_PATH=", os.environ.get("VDB_PATH"))
    _log(emit_raw, "[env]  CWD=", os.getcwd())

    items: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    errors: List[str] = list(state.get("errors", []))

    # 2) (변경) 리스트만 먼저 수집
    try:
        _log(emit_raw, "[crawl] list: nextunicorn_list start")
        items = __import__("asyncio").run(nextunicorn_list(limit=limit, headless=headless))
        _log(emit_raw, "[crawl] list: done; items=", len(items))
    except Exception as e:
        errors.append(str(e))
        _log(emit_raw, "[crawl] list ERROR:", e)
        traceback.print_exc()

    if emit_raw:
        print(json.dumps({"items": items, "errors": errors}, ensure_ascii=False, indent=2))

    # 3) 크로마: 존재여부 먼저 검사 → 존재하면 즉시 종료(break)
    pending: List[Dict[str, Any]] = []
    try:
        _log(emit_raw, "[chroma] START: get_company_store()")
        vectordb = get_vector_store()
        try:
            col_obj = getattr(vectordb, "_collection", None)
            _log(emit_raw, "[chroma] collection object exists =", bool(col_obj))
        except Exception:
            pass

        for idx, it in enumerate(items):
            name = (it.get("title") or "").strip()
            url = it.get("url") or ""
            _log(emit_raw, f"[loop:{idx}] name={name} url={url}")
            if not name:
                _log(emit_raw, f"[loop:{idx}] skip: empty name")
                continue

            _log(emit_raw, f"[loop:{idx}] find_company_exact_or_similar → start")
            found_id, _ = find_company_exact_or_similar(
                vectordb, company_name=name, k=3, score_threshold=0.18
            )
            _log(emit_raw, f"[loop:{idx}] find_company_exact_or_similar → found_id={found_id}")
            if found_id:
                # ✅ 정책: 하나라도 이미 있으면 즉시 중단
                _log(emit_raw, f"[loop:{idx}] exists → break (최신 우선 정책)")
                # 상태에 스킵 기록만 남기고 종료
                if emit_raw:
                    print(json.dumps(
                        {"chroma_created": [], "chroma_skipped": [{"name": name, "id": found_id, "reason": "exists"}]},
                        ensure_ascii=False, indent=2
                    ))
                return {
                    "limit": limit,
                    "headless": headless,
                    "emit_raw": emit_raw,
                    "items": items,
                    "details": details,  # 상세는 아직 안 받았으므로 빈 배열일 수 있음
                    "errors": errors,
                }

            pending.append({"title": name, "url": url})

    except Exception as e:
        errors.append(f"[chroma] {e}")
        _log(emit_raw, "[chroma] BLOCK ERROR:", e)
        traceback.print_exc()

    # 4) (존재하지 않는 것만) 상세 본문 수집 → LLM 정리/태깅 → 업서트
    try:
        if not pending:
            _log(emit_raw, "[detail] pending=0 → nothing to do")
        else:
            # 필요한 URL만 all-tab로 모아 배치 상세 수집
            urls = [ _normalize_all_tab(p["url"]) for p in pending if p.get("url") ]
            _log(emit_raw, f"[detail] batch fetch start; urls={len(urls)}")
            details = __import__("asyncio").run(
                nextunicorn_company_details_batch(urls, headless=headless)
            )
            _log(emit_raw, "[detail] batch fetch done; details=", len(details))

            # URL → full_text 매핑
            url2text = { d["url"].replace("?tab=all",""): d.get("full_text","") for d in details }

            created = []
            for idx, it in enumerate(pending):
                name = it["title"]
                url = it["url"]
                raw_text = url2text.get(url, "") or url2text.get(_normalize_all_tab(url), "")

                if not raw_text:
                    # 카드 정보만으로 최소 텍스트 구성
                    raw_text = "\n".join(filter(None, [name, url]))

                _log(emit_raw, f"[upsert:{idx}] LLM clean/tag start; name={name} raw_len={len(raw_text)}")
                cleaned, tags = _clean_and_tag_via_llm(name=name, raw_text=raw_text, hint="", emit=emit_raw)

                _log(emit_raw, f"[upsert:{idx}] upsert_company_profile start; tags={tags}")
                doc_id = upsert_company_profile(
                    vectordb,
                    company_name=name,
                    structured=cleaned,
                    url=url,
                    overwrite=False,
                    tags=tags,
                )
                _log(emit_raw, f"[upsert:{idx}] upsert_company_profile done; doc_id={doc_id}")
                created.append({"name": name, "id": doc_id, "tags": tags})

            if emit_raw:
                print(json.dumps({"chroma_created": created, "chroma_skipped": []},
                                 ensure_ascii=False, indent=2))
                try:
                    col = get_vector_store()._collection
                    cnt = col.count()
                    print("[chroma] companies.count =", cnt)
                except Exception as e:
                    print("[chroma] count error:", e)
                    traceback.print_exc()

    except Exception as e:
        errors.append(f"[detail] {e}")
        _log(emit_raw, "[detail] ERROR:", e)
        traceback.print_exc()

    # 5) 반환
    return {
        "limit": limit,
        "headless": headless,
        "emit_raw": emit_raw,
        "items": items,
        "details": details,
        "errors": errors,
    }

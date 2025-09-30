# agents/startup_search_agent.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import json
import re
import os
import random
import time
import traceback
from state import State

# ── 0) 프린트 로거 ────────────────────────────────────────────────────────────
def _log(*args):
    msg = " ".join(str(a) for a in args)
    print(f"[DEBUG] {msg}")

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

_COMBINED_PROMPT = ChatPromptTemplate.from_messages([
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
        "  }},\n"
        '  "tags": ["태그1","태그2","태그3"]\n'
        "}}\n\n"
        "[정리 규칙]\n"
        "1) cleaned 7키는 항상 포함.\n"
        "2) 해시태그/URL/광고·이벤트/내비게이션 제거, 중복 제거, 3+개행→1, 간단 맞춤법 보정.\n"
        "3) 각 섹션 최대 800자, 과장은 줄이고 사실 위주. 근거 없으면 빈 문자열(\"\").\n"
        "4) funding은 문서 어디에 있어도(소개/뉴스 등) 시리즈/라운드/누적 투자/투자 금액/투자자 신호를 모아 요약.\n\n"
        "[태그 규칙]\n"
        "한국어 2~4어절, 최대 3개. 핵심 비즈니스/가치/도메인 드러내기.\n\n"
        "[출력 형식]\n"
        "위 JSON만 출력(코드펜스 금지)."
    ),
    (
        "user",
        "회사명: {name}\n힌트(선택): {hint}\n\n원문 텍스트(raw_text):\n{raw_text}"
    ),
])

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

# ── 6) LLM: 정리+태깅 통합 호출 ──────────────────────────────────────────────
_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)

def _clean_and_tag_via_llm(
    name: str,
    raw_text: str,
    hint: str = "",
    *,
    emit: bool = True,  # 기본 True로 두고 항상 디버깅 출력
) -> Tuple[Dict[str, str], List[str]]:
    import json as _json

    _log("[LLM] combined: invoke start; raw_text_len:", len(raw_text), "name:", name)
    try:
        msgs = _COMBINED_PROMPT.format_messages(
            name=name,
            hint=hint or "",
            raw_text=raw_text,
        )
        out = _llm.invoke(msgs).content or ""
        _log("[LLM] combined: raw output head:", (out[:200].replace("\n", " ") + " ..."))

        # 코드펜스에 감싸오는 경우 대비
        m = _JSON_FENCE.search(out)
        if m:
            out = m.group(1)

        data = _json.loads(out) if out else {}
        cleaned_in = (data.get("cleaned") or {}) if isinstance(data, dict) else {}

        # cleaned 강제 보정 + 로컬 정리
        cleaned: Dict[str, str] = {}
        for k in ["summary", "services", "team", "funding", "news", "info", "company"]:
            v = str((cleaned_in.get(k) or "")).strip()
            v = _local_tidy(v)
            if len(v) > 800:
                v = v[:800]
            cleaned[k] = v

        # tags: 상위 레벨 배열 우선, 혹시 cleaned.tags 문자열로 줄 수도 있어 방어
        tags_raw = data.get("tags")
        tags: List[str] = []
        if isinstance(tags_raw, list):
            for t in tags_raw[:3]:
                if isinstance(t, str) and t.strip():
                    tags.append(t.strip())
        else:
            maybe_str = cleaned_in.get("tags")
            if isinstance(maybe_str, str) and maybe_str.strip():
                for t in maybe_str.split("|"):
                    tt = t.strip()
                    if tt:
                        tags.append(tt)
                tags = tags[:3]

        _log("[LLM] combined: done; lens:",
             {k: len(cleaned.get(k, "")) for k in cleaned},
             "tags:", tags)
        return cleaned, tags

    except Exception as e:
        _log("[LLM][ERROR]", e)
        traceback.print_exc()
        # 폴백: 전부 빈칸
        return (
            {
                k: ""
                for k in [
                    "summary",
                    "services",
                    "team",
                    "funding",
                    "news",
                    "info",
                    "company",
                ]
            },
            [],
        )

# ── 7) 메인 에이전트 ──────────────────────────────────────────────────────────
def startup_search_agent(state: State) -> State:
    """
    1) NextUnicorn 리스트만 먼저 수집
    2) VDB(Chroma)에서 회사명 존재 여부 확인
       - 하나라도 이미 있으면: 즉시 종료(최신 우선 정책)
       - 없으면: 해당 항목만 상세 본문 수집 → LLM 정리/태깅 → 업서트
    3) selected_companies 를 최대 10개로 채워 다음 노드가 사용할 수 있게 함
       - 이번에 업서트된 회사명이 있으면 그것들로
       - 없으면 기존 VDB에서 랜덤 10개 샘플링
    4) state 에 변경분만 덮어써서 반환
    """
    print("\n===== [START] startup_search_agent =====")
    t0 = time.time()

    # 0) 입력 파싱 및 기본 상태
    limit: int = state.get("limit") or _parse_limit_from_text(
        state.get("input_text"), default=2
    )
    headless: bool = state.get("headless", True)
    emit_raw: bool = state.get("emit_raw", False)

    _log("[BOOT] input_text=", state.get("input_text"))
    _log("[BOOT] limit=", limit, "headless=", headless)
    _log("[ENV ] VDB_PATH=", os.environ.get("VDB_PATH"))
    _log("[ENV ] CWD=", os.getcwd())

    items: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    errors: List[str] = list(state.get("errors", []))
    created_names: List[str] = []

    # 1) 리스트 수집
    try:
        _log("[CRAWL] nextunicorn_list: START")
        t_list0 = time.time()
        items = __import__("asyncio").run(
            nextunicorn_list(limit=limit, headless=headless)
        )
        t_list1 = time.time()
        _log("[CRAWL] nextunicorn_list: DONE items=", len(items),
             "elapsed=", f"{t_list1 - t_list0:.3f}s")
    except Exception as e:
        errors.append(str(e))
        _log("[CRAWL][ERROR]", e)
        traceback.print_exc()

    if emit_raw:
        print("[CRAWL] items dump:")
        print(json.dumps({"items": items, "errors": errors}, ensure_ascii=False, indent=2))

    # 2) 크로마 존재 여부 검사
    pending: List[Dict[str, Any]] = []
    vectordb = None
    try:
        _log("[CHROMA] get_vector_store: START")
        vectordb = get_vector_store()
        _log("[CHROMA] get_vector_store: DONE")
        try:
            col_obj = getattr(vectordb, "_collection", None)
            _log("[CHROMA] collection object exists =", bool(col_obj))
        except Exception:
            pass

        for idx, it in enumerate(items):
            name = (it.get("title") or "").strip()
            url = it.get("url") or ""
            _log(f"[CHROMA][LOOP {idx}] name={name} url={url}")

            if not name:
                _log(f"[CHROMA][LOOP {idx}] SKIP empty name")
                continue

            _log(f"[CHROMA][LOOP {idx}] find_company_exact_or_similar: START")
            found_id, _ = find_company_exact_or_similar(
                vectordb, company_name=name, k=3, score_threshold=0.18
            )
            _log(f"[CHROMA][LOOP {idx}] find_company_exact_or_similar: found_id={found_id}")

            if found_id:
                _log(f"[CHROMA][LOOP {idx}] EXISTS → EARLY EXIT (최신 우선 정책)")
                if emit_raw:
                    print(json.dumps(
                        {
                            "chroma_created": [],
                            "chroma_skipped": [{"name": name, "id": found_id, "reason": "exists"}],
                        },
                        ensure_ascii=False, indent=2
                    ))
                new_state = dict(state)
                new_state.update(
                    {
                        "limit": limit,
                        "headless": headless,
                        "emit_raw": emit_raw,
                        "items": items,
                        "details": details,
                        "errors": errors,
                    }
                )
                print(f"===== [END] startup_search_agent (early-exit, total {time.time()-t0:.3f}s) =====\n")
                return new_state

            pending.append({"title": name, "url": url})

        _log("[CHROMA] pending for details =", len(pending))

    except Exception as e:
        errors.append(f"[chroma] {e}")
        _log("[CHROMA][ERROR]", e)
        traceback.print_exc()

    # 3) (존재하지 않는 것만) 상세 본문 수집 → LLM 정리/태깅 → 업서트
    try:
        if not pending:
            _log("[DETAIL] pending=0 → nothing to do")
        else:
            urls = [_normalize_all_tab(p["url"]) for p in pending if p.get("url")]
            _log("[DETAIL] batch fetch: START urls=", len(urls))
            t_det0 = time.time()
            details = __import__("asyncio").run(
                nextunicorn_company_details_batch(urls, headless=headless)
            )
            t_det1 = time.time()
            _log("[DETAIL] batch fetch: DONE details=", len(details),
                 "elapsed=", f"{t_det1 - t_det0:.3f}s")

            # URL → full_text 매핑
            url2text = {d["url"].replace("?tab=all", ""): d.get("full_text", "") for d in details}

            created = []
            t_llm0 = time.time()
            for idx, it in enumerate(pending):
                name = it["title"]
                url = it["url"]
                raw_text = url2text.get(url, "") or url2text.get(_normalize_all_tab(url), "")
                if not raw_text:
                    raw_text = "\n".join(filter(None, [name, url]))

                _log(f"[UPSERT {idx}] LLM clean/tag: START name={name} raw_len={len(raw_text)}")
                cleaned, tags = _clean_and_tag_via_llm(
                    name=name, raw_text=raw_text, hint="", emit=True
                )
                _log(f"[UPSERT {idx}] LLM clean/tag: DONE tags={tags}")

                _log(f"[UPSERT {idx}] upsert_company_profile: START")
                doc_id = upsert_company_profile(
                    vectordb,
                    company_name=name,
                    structured=cleaned,
                    url=url,
                    overwrite=False,
                    tags=tags,
                )
                _log(f"[UPSERT {idx}] upsert_company_profile: DONE doc_id={doc_id}")
                created.append({"name": name, "id": doc_id, "tags": tags})

            t_llm1 = time.time()
            _log("[UPSERT] TOTAL LLM+UPSERT elapsed=", f"{t_llm1 - t_llm0:.3f}s")

            created_names = [c["name"] for c in created if "name" in c]

            if emit_raw:
                print("[UPSERT] dump:")
                print(json.dumps({"chroma_created": created, "chroma_skipped": []}, ensure_ascii=False, indent=2))
                try:
                    col = get_vector_store()._collection
                    cnt = col.count()
                    print(f"[CHROMA] investment_ai.count={cnt}")
                except Exception as e:
                    print("[CHROMA] count error:", e)
                    traceback.print_exc()

    except Exception as e:
        errors.append(f"[detail] {e}")
        _log("[DETAIL][ERROR]", e)
        traceback.print_exc()

    # 4) selected_companies 채우기 (업서트 성공분 or 기존 랜덤)
    try:
        if created_names:
            chosen = created_names[:10]
            _log("[SELECT] use created_names =", len(chosen))
        else:
            if vectordb is None:
                vectordb = get_vector_store()
            chosen = _sample_existing_companies(vectordb, n=10)
            _log("[SELECT] use sampled existing companies =", len(chosen))

        state.setdefault("selected_companies", [])
        state["selected_companies"] = chosen
        state.setdefault("current_company", None)
        state.setdefault("current_tags", [])
        state.setdefault("report_written", False)
        state.setdefault("investment_decision", None)
    except Exception as e:
        errors.append(f"[postselect] {e}")
        _log("[POSTSELECT][ERROR]", e)
        traceback.print_exc()

    # 5) 반환: 기존 state 복사 후 변경분 덮어쓰기
    new_state = dict(state)
    new_state.update(
        {
            "limit": limit,
            "headless": headless,
            "emit_raw": emit_raw,
            "items": items,
            "details": details,
            "errors": errors,
        }
    )

    total = time.time() - t0
    _log("[SUMMARY] items=", len(items), "details=", len(details), "errors=", len(errors))
    print(f"===== [END] startup_search_agent (total {total:.3f}s) =====\n")
    return new_state

# agents/startup_search_agent.py 내부, 유틸 아래에 추가
def _sample_existing_companies(vectordb, n: int = 10) -> List[str]:
    """
    Chroma 메타데이터(kind=company)에서 랜덤으로 최대 n개 회사명(name)을 샘플링.
    _collection.get(limit, offset)를 이용해 과도한 전체 로드를 피함.
    """
    names: List[str] = []
    try:
        col = getattr(vectordb, "_collection", None)
        if col is None:
            return names

        total = col.count()  # 전체 문서 수
        if total <= 0:
            return names

        # kind=company 만 집계 (최대 1000개까지 가져와서 샘플)
        res = col.get(where={"kind": "company"}, include=["metadatas", "ids"], limit=1000)
        metas = res.get("metadatas") or []
        ids = res.get("ids") or []
        pool = []
        for i, md in enumerate(metas):
            nm = (md or {}).get("name") or (ids[i] if i < len(ids) else None)
            if nm:
                pool.append(nm)

        if not pool:
            return names

        if len(pool) <= n:
            return pool

        return random.sample(pool, n)
    except Exception:
        # 문제가 생기면 peek로 대체 (처음 N개)
        try:
            col = getattr(vectordb, "_collection", None)
            if col is None:
                return names
            peek = col.peek(limit=n)
            metas = peek.get("metadatas") or []
            ids = peek.get("ids") or []
            out = []
            for i, md in enumerate(metas):
                nm = (md or {}).get("name") or (ids[i] if i < len(ids) else None)
                if nm:
                    out.append(nm)
            return out
        except Exception:
            return names

# ── 8) 단독 실행 ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    s: State = {
        "input_text": "NextUnicorn에서 스타트업 2개 알려줘",
        "headless": True,
        "emit_raw": True,
    }
    print("[MAIN] invoke startup_search_agent")
    try:
        s = startup_search_agent(s)
        print("[MAIN] done")
    except Exception as e:
        print("[MAIN][ERROR]", e)
        traceback.print_exc()

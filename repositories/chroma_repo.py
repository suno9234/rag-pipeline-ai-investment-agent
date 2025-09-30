# repositories/chroma_repo.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import re

from langchain_chroma import Chroma
from config.chroma import get_company_store, get_industry_store
from tools.nextunicorn import summarize_company_text  # 상세 본문 → 섹션 요약


def _slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^0-9a-z가-힣]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "doc"


def _build_company_doc_text(company_name: str, structured: Dict[str, str]) -> str:
    parts = [
        f"[회사명] {company_name}",
        f"[요약] {structured.get('summary','').strip()}",
        f"[서비스/제품] {structured.get('services','').strip()}",
        f"[팀] {structured.get('team','').strip()}",
        f"[투자] {structured.get('funding','').strip()}",
        f"[소식] {structured.get('news','').strip()}",
        f"[회사 정보] {structured.get('info','').strip()}",
        f"[원문 머리] {structured.get('company','').strip()}",
    ]
    return "\n".join([p for p in parts if p and p.strip()])


# repositories/chroma_repo.py


def find_company_exact_or_similar(store, company_name: str, k=3, score_threshold=0.18):
    # 0) 정규화
    q = (company_name or "").strip()
    if not q:
        return None, None

    norm = q.lower().replace(" ", "")

    # 1) 메타데이터 exact(where)로 먼저 조회
    try:
        col = store._collection  # langchain_chroma.Chroma 내부의 chromadb 컬렉션
        got = col.get(
            where={
                "$or": [
                    {"name": {"$eq": q}},
                    {"name": {"$eq": norm}},
                    {"id": {"$eq": q}},
                    {"id": {"$eq": norm}},
                ]
            },
            include=["metadatas"],
        )
        ids = got.get("ids") or []
        if ids:
            return ids[0], 1.0  # 확정 매치
    except Exception:
        pass

    # 2) 임베딩 유사도(문서 본문 기반)
    try:
        sims = store.similarity_search_with_score(q, k=k)
        # sims: [(Document, score), ...]  # langchain_chroma는 score가 L2 거리일 수 있음
        # score 의미가 거리면 작을수록 유사 → threshold 로직 맞게 해석 필요
        best = None
        for doc, score in sims:
            md = doc.metadata or {}
            name = (md.get("name") or "").strip()
            if name and (name == q or name.lower().replace(" ", "") == norm):
                best = (md.get("id") or name, score)
                break
        if not best and sims:
            # 이름이 딱 안맞아도, 임계치 안쪽이면 채택
            doc, score = sims[0]
            if score is not None and score <= score_threshold:
                md = doc.metadata or {}
                return (md.get("id") or md.get("name")), score
        if best:
            return best
    except Exception:
        pass

    return None, None


def upsert_company_profile(
    store, company_name, structured, url=None, overwrite=False, tags=None
) -> str:
    doc_id = _slugify(company_name)
    text = _build_company_doc_text(company_name, structured)
    tags_str = " | ".join(
        t.strip() for t in (tags or []) if isinstance(t, str) and t.strip()
    )
    metadata = {
        "kind": "company",
        "id": doc_id,
        "name": company_name,
        "url": url or "",
        "tags": tags_str,  # ← 항상 문자열 (빈 리스트면 빈 문자열)
    }
    if overwrite:
        try:
            store.delete(ids=[doc_id])
        except Exception:
            pass
    store.add_texts(texts=[text], metadatas=[metadata], ids=[doc_id])
    return doc_id


def ensure_company_profiles(
    items: List[Dict[str, Any]],
    details: List[Dict[str, Any]],
    *,
    overwrite: bool = False,
) -> Dict[str, Any]:
    store = get_company_store()
    url2detail = {d.get("url"): d for d in (details or [])}
    created, skipped, errors = [], [], []

    for it in items or []:
        name = (it.get("title") or "").strip()
        url = it.get("url")
        if not name:
            continue

        try:
            found_id, _ = find_company_exact_or_similar(store, name)
            if found_id and not overwrite:
                skipped.append({"name": name, "id": found_id, "reason": "exists"})
                continue

            structured = {}
            if url and url in url2detail and url2detail[url].get("full_text"):
                structured = summarize_company_text(url2detail[url]["full_text"])
            else:
                structured = {
                    "summary": it.get("summary", ""),
                    "services": "",
                    "team": "",
                    "news": "",
                    "funding": "",
                    "info": "",
                    "company": "",
                }

            # 기본 ensure는 태그 없이 저장 (태그는 에이전트에서 생성해 넘겨주는 걸 권장)
            doc_id = upsert_company_profile(
                store, name, structured, url=url, overwrite=overwrite, tags=None
            )
            created.append({"name": name, "id": doc_id})

        except Exception as e:
            errors.append(f"{name}: {e}")

    try:
        store.persist()
    except Exception:
        pass

    return {"created": created, "skipped": skipped, "errors": errors}


def upsert_industry_report(
    sector: str,
    title: str,
    body: str,
    source_url: Optional[str] = None,
) -> str:
    store = get_industry_store()
    doc_id = _slugify(f"{sector}-{title[:60]}")
    metadata = {
        "kind": "industry",
        "sector": sector,
        "title": title,
        "source": source_url or "",
    }
    store.add_texts(texts=[body], metadatas=[metadata], ids=[doc_id])
    try:
        store.persist()
    except Exception:
        pass
    return doc_id


def query_industry_by_sector(sector: str, k: int = 5):
    store = get_industry_store()
    return store.similarity_search(
        query=sector, k=k, filter={"kind": "industry", "sector": sector}
    )


def get_company_by_name(
    company_name: str,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    store = get_company_store()
    return find_company_exact_or_similar(store, company_name)

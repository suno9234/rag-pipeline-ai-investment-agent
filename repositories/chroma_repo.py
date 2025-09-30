# repositories/chroma_repo.py
from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional

# ✅ 더 이상 config.chroma 에서 아무 것도 import 하지 않습니다.
# from config.chroma import get_company_store, get_industry_store  # ← 삭제

# LangChain Chroma VectorStore를 받아서만 동작하도록 구성합니다.
# vectordb 타입: langchain_chroma.Chroma

def _join_sections(structured: Dict[str, str]) -> str:
    parts = []
    def add(label: str, key: str):
        v = (structured.get(key) or "").strip()
        if v:
            parts.append(f"[{label}] {v}")
    add("회사명", "company")
    add("요약", "summary")
    add("서비스/제품", "services")
    add("팀", "team")
    add("투자", "funding")
    add("소식", "news")
    add("회사 정보", "info")
    return "\n".join(parts).strip()

def find_company_exact_or_similar(
    vectordb,
    *,
    company_name: str,
    k: int = 3,
    score_threshold: float = 0.18,
) -> Tuple[Optional[str], float]:
    """
    회사명이 동일한 문서가 있으면 그 id를, 없으면 유사검색으로 근접 id를 반환.
    score_threshold는 (거리 기반) 점수의 최대 허용치로 가정합니다.
    """
    # 1) 메타데이터 exact match 시도
    try:
        # LangChain의 Chroma는 내부 collection에 접근 가능
        col = getattr(vectordb, "_collection", None)
        if col is not None:
            res = col.get(
                where={"$and": [{"kind": "company"}, {"name": company_name}]},
                include=["metadatas", "ids"],
            )
            ids = (res.get("ids") or [])
            metas = (res.get("metadatas") or [])
            if ids:
                return ids[0], 0.0
    except Exception:
        pass

    # 2) 유사도 검색 (회사명 쿼리)
    try:
        # LC 버전에 따라 함수가 다를 수 있어 두 가지를 시도
        docs_scores = None
        try:
            docs_scores = vectordb.similarity_search_with_score(
                query=company_name,
                k=k,
                filter={"kind": "company"},
            )
        except Exception:
            # 일부 버전에선 relevance_scores 형태 사용
            docs_scores = vectordb.similarity_search_with_relevance_scores(
                query=company_name,
                k=k,
                filter={"kind": "company"},
            )
        if not docs_scores:
            return None, 1.0

        # (Document, score) or (Document, relevance) 형태
        best_id = None
        best_score = 1e9
        for item in docs_scores:
            # tuple 형태 가정
            if isinstance(item, tuple) and len(item) >= 2:
                doc, score = item[0], float(item[1])
            else:
                # 혹시 다른 형태면 skip
                continue
            meta = getattr(doc, "metadata", {}) or {}
            cid = meta.get("id") or meta.get("name")
            if cid is None:
                continue
            if score < best_score:
                best_score = score
                best_id = cid

        if best_id is not None and best_score <= score_threshold:
            return best_id, best_score
        return None, best_score
    except Exception:
        return None, 1.0


def upsert_company_profile(
    vectordb,
    *,
    company_name: str,
    structured: Dict[str, str],
    url: str,
    overwrite: bool = False,
    tags: List[str] | None = None,
) -> str:
    """
    단일 문서를 upsert.
    - id 는 company_name 그대로 사용 (한글 가능)
    - tags(list)는 " | " 로 합쳐 메타데이터에 저장
    """
    doc_id = company_name
    meta: Dict[str, Any] = {
        "id": company_name,
        "name": company_name,
        "url": url,
        "kind": "company",
        "tags": " | ".join(tags or []),
    }
    text = _join_sections(structured)

    # 덮어쓰기 옵션
    if overwrite:
        try:
            vectordb.delete(ids=[doc_id])
        except Exception:
            pass

    # 이미 존재하면 덮어쓰지 않고 종료 (overwrite=False)
    if not overwrite:
        try:
            col = getattr(vectordb, "_collection", None)
            if col is not None:
                res = col.get(
                    where={"$and": [{"kind": "company"}, {"id": doc_id}]},
                    include=["ids"],
                )
                if res and res.get("ids"):
                    return doc_id
        except Exception:
            pass

    # upsert (LangChain VectorStore API)
    try:
        vectordb.add_texts(
            texts=[text],
            metadatas=[meta],
            ids=[doc_id],
        )
    except Exception:
        # add_texts에서 중복으로 실패하면 delete 후 재시도
        try:
            vectordb.delete(ids=[doc_id])
        except Exception:
            pass
        vectordb.add_texts(
            texts=[text],
            metadatas=[meta],
            ids=[doc_id],
        )

    # 영속화는 Chroma가 자동 처리 (langchain_chroma는 persist_directory 지정 시 내부적으로 flush)
    return doc_id

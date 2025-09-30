from __future__ import annotations
from pathlib import Path
import os, sys, json
from typing import List, Dict

# 경로 설정
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# 외부 모듈
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.chroma import get_vector_store
from agents.industry_search import run_search, get_default_config

DOCS_DIR = project_root / "docs"
DOCS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# VDB 조회/삽입 유틸
# ─────────────────────────────────────────────────────────────
def _chroma_exists_by_tag(tag: str) -> bool:
    """Chroma 컬렉션 where 필터로 tag가 있는지 간단 점검."""
    try:
        vdb = get_vector_store()
        col = vdb._collection  # langchain_chroma 내부 chromadb collection
        res = col.get(where={"kind": "industry"}, limit=1)
        return bool(res and res.get("ids"))
    except Exception:
        return False

def _chroma_insert_texts(texts: List[str]):
    """
    메타데이터:
      - kind: "industry"
    """
    vdb = get_vector_store()
    metas = [{"kind": "industry"} for _ in texts]
    vdb.add_texts(texts=texts, metadatas=metas)

# ─────────────────────────────────────────────────────────────
# 결과(JSON) → 청크 → MD 저장 → VDB 적재
# ─────────────────────────────────────────────────────────────
def _load_results(json_path: Path) -> Dict[str, List[Dict]]:
    return json.loads(json_path.read_text(encoding="utf-8"))

def _build_markdown(group: str, items: List[Dict]) -> str:
    lines = [f"# {group}\n"]
    for it in items:
        title = (it.get("title") or "").strip()
        url = (it.get("url") or "").strip()
        content = (it.get("content") or "").strip()
        if title:
            lines.append(f"## {title}")
        if url:
            lines.append(f"- Source: {url}")
        if content:
            lines.append("\n" + content + "\n")
        lines.append("\n---\n")
    return "\n".join(lines)

def _split_markdown(md_text: str, chunk_size=1000, chunk_overlap=150) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [c.page_content for c in splitter.create_documents([md_text])]

def _persist_group_md(industry: str, group: str, md_text: str) -> Path:
    md_path = DOCS_DIR / f"{industry}_{group}.md"
    md_path.write_text(md_text, encoding="utf-8")
    return md_path

# ─────────────────────────────────────────────────────────────
# LangGraph 노드: industry_embedding
# ─────────────────────────────────────────────────────────────
def industry_embedding(
    state: dict,
    *,
    force: bool = False,
    use_global_sources: bool = False,
) -> dict:
    """
    LangGraph add_node(...)로 바로 연결 가능한 노드 함수.

    - industry / groups / synonyms 는 search 모듈의 하드코딩 값을 사용
      → get_default_config()로 읽어온다.
    - VDB(Chroma)에 kind="industry" 가 이미 있으면 해당 그룹 임베딩 PASS
    - 없으면:
        (1) 검색 실행 → docs/{industry}_search_results.json 저장
        (2) 결과를 MD로 합치고 저장
        (3) 청크 후 Chroma에 add_texts(texts, metadatas=...)로 적재
          * 메타데이터: kind="industry"
    - 반환: 입력 state 그대로 (노드 내부용 정보는 state에 넣지 않음)
    """
    industry, groups, _syns = get_default_config()
    out_json_path = DOCS_DIR / f"{industry}_search_results.json"

    # 1) 그룹별로 VDB 존재 여부 확인 (검색 전에 PASS 판단)
    to_process = []
    for g in groups:
        if not force and _chroma_exists_by_tag(g):
            print(f"⏭️  [SKIP] '{g}' → VDB(kind=industry)에 이미 존재")
            continue
        to_process.append(g)

    if not to_process:
        print("✅ 모든 그룹이 이미 VDB에 존재합니다. (검색/임베딩 생략)")
        return state

    # 2) 검색 실행 (synonyms은 search 모듈 하드코딩 사용 → override 전달 안 함)
    print(f"🔎 검색 실행: industry={industry}, groups={to_process}, use_global={use_global_sources}")
    out_json_path = run_search(
        industry=industry,
        groups=to_process,
        out_json=out_json_path,
        use_global=use_global_sources,
        synonyms_override=None,
    )

    # 3) 결과 로드
    results = _load_results(out_json_path)

    # 4) 그룹별로 MD 작성 → 청크 → VDB 적재
    for g in to_process:
        items = results.get(g, [])
        if not items:
            print(f"⚠️  [{g}] 검색 결과가 비어 있어 임베딩을 건너뜁니다.")
            continue

        md_text = _build_markdown(g, items)
        md_path = _persist_group_md(industry, g, md_text)
        chunks = _split_markdown(md_text)

        if not chunks:
            print(f"⚠️  [{g}] 청크 생성 실패(빈 문서). 건너뜁니다.")
            continue

        print(f"🧩 [{g}] 청크 {len(chunks)}개 → VDB 적재 (kind=industry)")
        _chroma_insert_texts(chunks)

    print("🎉 임베딩 파이프라인 완료")
    return state

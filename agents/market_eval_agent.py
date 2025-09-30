# agents/market_eval_agent.py
from __future__ import annotations
from typing import List, TypedDict, Optional, Iterable, Dict, Any
from dataclasses import dataclass

# LangChain / OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

# RAG 도구: 우선 tools.rag에서 가져오고, 없으면 Chroma 직접 사용(fallback)
try:
    from tools.rag import retrieve_docs  # 기대 시그니처: retrieve_docs(query: str, where: dict | None, k: int) -> List[Document]
    _HAS_RAG_TOOL = True
except Exception:
    _HAS_RAG_TOOL = False
    # fallback: config.chroma 사용
    from config.chroma import get_vector_store  # get_vector_store() -> Chroma
    try:
        from chromadb.utils import embedding_functions  # type: ignore
    except Exception:
        pass


# ==============================
# Graph State (7개 필드 고정) - graph.py의 initial_state 키와 일치
# ==============================
class GraphState(TypedDict):
    selected_companies: List[str]     # 선정 기업 10개 기업명 목록 (분석 완료 시 pop은 상위 단계에서)
    current_company: str              # 현재 기업명
    current_tags: List[str]           # 현재 기업의 태깅 목록
    market_analysis: Optional[str]    # 시장성 조사 내용 (텍스트) - 이 에이전트에서 "업데이트"만 함
    competitor_analysis: Optional[str]# 경쟁사 분석 내용 (텍스트) - 변경하지 않음
    report_written: bool              # 보고서 작성 여부 (상위 단계에서 관리)
    investment_decision: Optional[bool]  # 투자 여부 판단 결과 (상위 단계에서 결정)


# ==============================
# LLM Structured Output Schema (= Grade)
# ==============================
class Grade(BaseModel):
    # 0~2 정수 점수 3종 + 근거 설명
    score_market_size: int = Field(..., ge=0, le=2, description="시장 크기 0~2")
    score_growth: int      = Field(..., ge=0, le=2, description="성장 가능성 0~2")
    score_demand: int      = Field(..., ge=0, le=2, description="고객 수요 0~2")
    rationale: str         = Field(..., description="간단한 평가 근거 (자연어)")


# ==============================
# 내부 유틸: RAG 질의 파라미터
# ==============================
@dataclass
class RagQuery:
    query: str
    where: Optional[Dict[str, Any]] = None
    k: int = 5


# ==============================
# RAG: 도큐먼트 → 텍스트 병합
# ==============================
def _concat_docs_text(docs: Iterable[Any], max_chars: int = 3000) -> str:
    """RAG 문서들의 page_content를 이어 붙여 프롬프트에 넣을 문자열로 만든다."""
    parts: List[str] = []
    total = 0
    for d in docs:
        content = getattr(d, "page_content", "") or ""
        if not content:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        snippet = content[:remaining]
        parts.append(snippet)
        total += len(snippet)
    return "\n\n---\n\n".join(parts)


# ==============================
# RAG: 산업/기업 컨텍스트 수집
#  - data_type == "산업" AND tag ∈ current_tags → context_industry
#  - data_type == "기업" AND tag ∈ current_tags AND source == current_company → context
# ==============================
def _build_rag_contexts(*, company: str, tags: List[str], k_industry: int = 5, k_company: int = 5) -> dict:
    # where 필터 구성
    where_industry = {"kind": "industry"}

    where_company = {"kind": "company", "name": company}

    # 질의어(간단): 태그/회사명 기반
    query_industry = ", ".join(tags) if tags else company
    query_company = company

    # RAG 실행 (우선 tools.rag, 없으면 Chroma 직접)
    if _HAS_RAG_TOOL:
        ind_docs = retrieve_docs(query=query_industry, where=where_industry, k=k_industry)
        com_docs = retrieve_docs(query=query_company, where=where_company, k=k_company)
    else:
        vectordb = get_vector_store()
        ind_docs = vectordb.similarity_search(query_industry, k=k_industry, where=where_industry)
        com_docs = vectordb.similarity_search(query_company, k=k_company, where=where_company)

    return {
        "context_industry": _concat_docs_text(ind_docs),
        "context_company": _concat_docs_text(com_docs),
    }


# ==============================
# 메인: 시장성 평가 에이전트
#  - PromptTemplate 사용 (context, question, context_industry)
#  - LLM structured output(Grade) 강제
#  - 결과는 market_analysis 상단에 헤더로 삽입
# ==============================
def market_eval_agent(state: GraphState, *, model_name: str = "gpt-4o-mini") -> GraphState:
    """
    - state에서 현재 기업명/태그를 읽고,
    - RAG로 산업/기업 컨텍스트를 수집한 뒤,
    - PromptTemplate(context, question, context_industry)로 LLM에 평가 요청,
    - 응답(Grade)을 market_analysis 상단 헤더로 삽입하여 state를 반환한다.
    - 나머지 6개 필드는 변경하지 않는다.
    """
    company = (state.get("current_company") or "").strip()
    base_market_text = state.get("market_analysis") or ""  # 기존 텍스트 보존

    # 1) RAG 컨텍스트 구축
    rag_ctx = _build_rag_contexts(company=company, tags=tags)
    context_industry = rag_ctx["context_industry"]
    context_company = rag_ctx["context_company"]

    # 2) LLM 및 구조화 출력 준비
    #    예시 요구: model = ChatOpenAI(temperature=0, model=MODEL_NAME, streaming=True)
    model = ChatOpenAI(temperature=0, model=model_name, streaming=True)
    llm_with_tool = model.with_structured_output(Grade, method="function_calling")

    # 3) 프롬프트 템플릿 (요구 포맷에 맞춤)
    #    template는 {context}, {question}, {context_industry} 세 변수를 사용
    prompt = PromptTemplate(
        template=(
            "너는 벤처투자 심사역이다. 주어진 정보를 바탕으로 시장성을 평가하라.\n\n"
            "평가 항목(정수, 0~2):\n"
            "- score_market_size: 0~2 (2=현재도 크고 앞으로도 커질 가능성 높음, "
            "1=현재 성장 중이나 미래는 불확실, 0=작거나 제한적)\n"
            "- score_growth: 0~2 (2=높음, 1=보통, 0=낮음)\n"
            "- score_demand: 0~2 (2=뚜렷, 1=보통, 0=낮음)\n"
            "- rationale: 간단한 평가 근거 (자연어 설명)\n\n"
            "반드시 위 4개 키만 포함된 JSON으로 답하라.\n\n"
            "[기업명]\n"
            "{question}\n\n"
            "[기업 관련 컨텍스트]\n"
            "{context}\n\n"
            "[산업 관련 컨텍스트]\n"
            "{context_industry}\n"
        ),
        input_variables=["context", "question", "context_industry"],
    )

    # 4) 프롬프트 렌더링
    formatted_prompt = prompt.format(
        context=context_company,          # 기업 RAG 컨텍스트
        question=company,                 # 현재 기업명
        context_industry=context_industry # 산업 RAG 컨텍스트
    )

    # 5) LLM 호출 (구조화 출력 Grade)
    out: Grade = llm_with_tool.invoke(formatted_prompt)

    # 6) 헤더 블록 만들기 (market_analysis 상단에 삽입)
    header = (
        "=== Market Evaluation ===\n"
        f"시장크기: {out.score_market_size}\n"
        f"성장가능성: {out.score_growth}\n"
        f"고객수요: {out.score_demand}\n"
        f"근거: {out.rationale}\n"
        "=== /Market Evaluation ==="
    )
    new_market_text = f"{header}\n\n{base_market_text}" if base_market_text else header

    # 7) 동일한 7개 상태로 반환 (오직 market_analysis만 업데이트)
    return GraphState(
        selected_companies=state["selected_companies"],
        current_company=state["current_company"],
        current_tags=state["current_tags"],
        market_analysis=new_market_text,
        competitor_analysis=state["competitor_analysis"],
        report_written=state["report_written"],
        investment_decision=state["investment_decision"],
    )

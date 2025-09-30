from __future__ import annotations
from typing import List, Dict, Any
import traceback

from state import State
from config.chroma import get_vector_store
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# LLM 초기화
_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# 경쟁사 분석 프롬프트
_COMPETITOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 스타트업 투자 분석 전문가입니다. "
            "주어진 기업과 경쟁사들의 정보를 바탕으로 경쟁사 분석 보고서를 작성해주세요.\n\n"
            "## 분석 대상 기업\n"
            "기업명: {current_company}\n"
            "분석 내용: {current_analysis}\n\n"
            "## 경쟁사 정보 (벡터 유사도 기준 선별)\n"
            "{competitors_text}\n\n"
            "## 분석 요청사항\n"
            "다음 항목들을 포함하여 경쟁사 분석 보고서를 작성해주세요:\n"
            "1. **경쟁 환경 개요**\n"
            "   - 주요 경쟁사 현황 (벡터 유사도 기준)\n"
            "   - 시장 내 경쟁 강도 분석\n"
            "   - 비즈니스 내용 유사도를 통한 경쟁 구도 파악\n"
            "2. **경쟁사별 상세 분석**\n"
            "   - 각 경쟁사의 핵심 특징 및 차별화 포인트\n"
            "   - 사업 모델 및 수익 구조 분석\n"
            "   - 벡터 유사도를 고려한 경쟁 강도 평가\n"
            "3. **경쟁 우위 분석**\n"
            "   - 분석 대상 기업의 차별화 포인트\n"
            "   - 경쟁사 대비 강점 및 약점\n"
            "   - 비즈니스 내용 유사도 영역에서의 경쟁력 평가\n"
            "4. **시장 포지셔닝**\n"
            "   - 시장 내 위치 분석 (내용 기반 유사도)\n"
            "   - 경쟁 전략 제안\n"
            "   - 시장 진입 장벽 및 기회 요소\n"
            "5. **투자 관점에서의 평가**\n"
            "   - 경쟁 환경이 투자에 미치는 영향\n"
            "   - 벡터 유사도가 높은 경쟁사들의 위험도 평가\n"
            "   - 리스크 요인 및 기회 요소\n"
            "   - 투자 결정에 필요한 핵심 고려사항\n\n"
            "분석 보고서는 구체적이고 실용적인 인사이트를 제공해야 하며, "
            "벡터 유사도 정보를 활용하여 투자 결정에 도움이 되는 정보를 포함해야 합니다."
        )
    ]
)


def competitor_analysis_agent(state: State) -> State:
    """
    현재 회사 정보를 기반으로 벡터DB에서 유사 기업을 검색하여
    LLM에게 경쟁사 분석 보고서를 생성하도록 요청.
    """
    try:
        vectordb = get_vector_store()
        current_company = state.get("current_company")

        if not current_company:
            state["competitor_analysis"] = "⚠️ current_company 없음"
            return state

        # 1) 현재 회사 문서 검색
        docs = vectordb.similarity_search(
            current_company,
            k=1,
            filter={"kind": "company", "name": current_company},
        )
        current_analysis = docs[0].page_content if docs else "⚠️ 현재 기업 정보 없음"

        # 2) 경쟁사 후보 검색 (자기 자신 제외)
        competitors = vectordb.similarity_search(
            current_analysis,
            k=3,
            filter={"kind": "company"},
        )
        competitors = [c for c in competitors if c.metadata.get("name") != current_company]

        competitors_text = "\n\n".join(
            f"- {c.metadata.get('name')}\n{c.page_content[:500]}..."
            for c in competitors
        ) or "⚠️ 경쟁사 없음"

        # 3) LLM 분석 요청
        msgs = _COMPETITOR_PROMPT.format_messages(
            current_company=current_company,
            current_analysis=current_analysis,
            competitors_text=competitors_text,
        )
        output = _llm.invoke(msgs).content or ""

        # 4) state 업데이트
        state["competitor_analysis"] = output.strip()

    except Exception as e:
        traceback.print_exc()
        state["competitor_analysis"] = f"⚠️ 경쟁사 분석 실패: {e}"

    return state

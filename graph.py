from langgraph.graph import END, StateGraph
from state import State
from config.chroma import get_vector_store

# === Agent import ===
from agents.startup_search_agent import startup_search_agent
from agents.industry_search import industry_search_agent
from agents.market_eval_agent import market_eval_agent
from agents.competitor_analysis_agent import competitor_analysis_agent
from agents.investment_decision_agent import investment_decision_agent
from agents.report_writer_agent import report_writer_agent


# === Resume Analysis Node ===
def resume_analysis_node(state: State) -> State:
    """
    기업 리스트와 보고서 작성 여부를 기반으로 상태만 갱신하고 State를 반환
    (분기 결정은 route_resume_analysis에서 수행)
    """
    if state.get("selected_companies"):
        # 1) 다음 기업 하나 꺼내기 (dict/list 방어)
        current_item = state["selected_companies"].pop(0)
        if isinstance(current_item, dict):
            current = (current_item.get("title") or "").strip()
        else:
            current = str(current_item or "").strip()

        state["current_company"] = current

        # 2) VDB에서 현재 기업명으로 검색 → 태그 파싱 (where 미사용, 로컬 필터링)
        tags = []
        if current:
            try:
                vectordb = get_vector_store()
                raw_docs = vectordb.similarity_search(query=current, k=32)

                def _norm(s: str) -> str:
                    # 공백 제거 후 비교: "제이 카" == "제이카"
                    return "".join((s or "").split())

                # kind=company && name == current(정규화 비교)
                filtered = [
                    d for d in raw_docs
                    if d.metadata.get("kind") == "company"
                    and _norm(d.metadata.get("name", "")) == _norm(current)
                ]

                if filtered:
                    tags_str = filtered[0].metadata.get("tags", "") or ""
                    tags = [t.strip() for t in tags_str.split("|") if t.strip()]
            except Exception as e:
                # 검색 실패해도 파이프라인 계속
                print(f"[resume_analysis] VDB lookup failed for {current}: {e}")

        state["current_tags"] = tags
    # state만 반환 (여기서 분기 라벨을 반환하면 안 됨)
    return state


def route_resume_analysis(state: State) -> str:
    """
    상태를 기준으로 다음 단계 라벨만 반환
    """
    if state.get("current_company"):
        # 방금 설정된 회사가 있으니 평가로 진행
        return "market_eval"
    # 더 이상 분석할 기업이 없을 때
    if state.get("report_written"):
        return "end"
    return "report_writer"


# === Investment Decision Routing ===
def route_investment_decision(state: State) -> str:
    return "report_writer" if state.get("investment_decision") else "resume_analysis"


# === Workflow 정의 ===
workflow = StateGraph(State)

# === 노드 등록 ===
workflow.add_node("startup_search", startup_search_agent)
workflow.add_node("industry_search", industry_search_agent)
workflow.add_node("resume_analysis", resume_analysis_node)   # ← 노드는 State 반환
workflow.add_node("market_eval", market_eval_agent)
workflow.add_node("competitor_analysis", competitor_analysis_agent)
workflow.add_node("investment_decision", investment_decision_agent)
workflow.add_node("report_writer", report_writer_agent)

# === 시작점 ===
workflow.set_entry_point("startup_search")

# === 노드 연결 ===
workflow.add_edge("startup_search", "industry_search")
workflow.add_edge("industry_search", "resume_analysis")

# resume_analysis → 분기
workflow.add_conditional_edges(
    "resume_analysis",
    route_resume_analysis,   # ← 문자열(라벨)만 반환하는 라우터
    {
        "market_eval": "market_eval",
        "report_writer": "report_writer",
        "end": END,
    }
)

workflow.add_edge("market_eval", "competitor_analysis")
workflow.add_edge("competitor_analysis", "investment_decision")

workflow.add_conditional_edges(
    "investment_decision",
    route_investment_decision,
    {
        "report_writer": "report_writer",
        "resume_analysis": "resume_analysis",
    }
)

# 보고서 작성 후 다시 resume_analysis
workflow.add_edge("report_writer", "resume_analysis")

investment_app = workflow.compile()

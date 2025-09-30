from langgraph.graph import END, StateGraph
from state import State
from config.chroma import get_vector_store

# === Agent import ===
from agents.startup_search_agent import startup_search_agent
from agents.industry_search_agent import industry_search_agent
from agents.market_eval_agent import market_eval_agent
from agents.competitor_analysis_agent import competitor_analysis_agent
from agents.investment_decision_agent import investment_decision_agent
from agents.report_writer_agent import report_writer_agent

# === Resume Analysis Node ===
def resume_analysis(state: State) -> str:
    """
    기업 리스트와 보고서 작성 여부를 기반으로 다음 단계 결정
    """
    if state["selected_companies"]:  # 아직 분석할 기업 남음
        current = state["selected_companies"].pop(0)
        state["current_company"] = current

        # VDB에서 현재 기업명으로 검색 → 태그 가져오기
        vectordb = get_vector_store()
        results = vectordb.similarity_search(
            query=current,  # 기업명 기준 검색
            k=1,           # 기업은 유일하니까 1개만 가져오기
            filter={"name": current, "kind": "company"}  # 메타데이터 조건
        )

        if results:
            # ✅ tags는 문자열이므로 split 처리
            tags_str = results[0].metadata.get("tags", "")
            tags = [t.strip() for t in tags_str.split("|") if t.strip()]
            state["current_tags"] = tags
        else:
            state["current_tags"] = []
            
        return "market_eval"
    else:
        if state["report_written"]:  # 보고서 하나라도 있음 → 종료
            return "end"
        else:  # 보고서가 전혀 없음 → 사유 보고서 작성
            return "report_writer"

# === Investment Decision Routing ===
def route_investment_decision(state: State) -> str:
    """
    투자 여부 결과에 따라 다음 단계 결정
    """
    if state["investment_decision"]:  # 투자 O → 보고서 작성
        return "report_writer"
    else:  # 투자 X → 바로 다음 기업 진행
        return "resume_analysis"

# === Workflow 정의 ===
workflow = StateGraph(State)

# === 노드 등록 ===
workflow.add_node("startup_search", startup_search_agent)
workflow.add_node("industry_search", industry_search_agent)
workflow.add_node("resume_analysis", resume_analysis)
workflow.add_node("market_eval", market_eval_agent)
workflow.add_node("competitor_analysis", competitor_analysis_agent)
workflow.add_node("investment_decision", investment_decision_agent)
workflow.add_node("report_writer", report_writer_agent)

# === 시작점 ===
workflow.set_entry_point("startup_search")

# === 노드 연결 ===
workflow.add_edge("startup_search", "industry_search")
workflow.add_edge("industry_search", "resume_analysis")

workflow.add_conditional_edges(
    "resume_analysis",
    lambda state: resume_analysis(state),
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
    lambda state: route_investment_decision(state),
    {
        "report_writer": "report_writer",
        "resume_analysis": "resume_analysis",
    }
)

# 보고서 작성 후 다시 resume_analysis
workflow.add_edge("report_writer", "resume_analysis")

investment_app = workflow.compile()
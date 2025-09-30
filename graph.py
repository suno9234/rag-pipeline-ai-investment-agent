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

# === 로그 래퍼 함수들 ===
def logged_startup_search(state: State) -> State:
    print("\n🚀 [STARTUP_SEARCH] 시작 - 스타트업 검색 중...")
    result = startup_search_agent(state)
    companies = result.get("selected_companies", [])
    print(f"✅ [STARTUP_SEARCH] 완료 - {len(companies)}개 기업 선정: {companies[:3]}{'...' if len(companies) > 3 else ''}")
    return result

def logged_industry_search(state: State) -> State:
    print("\n🏭 [INDUSTRY_SEARCH] 시작 - 산업 분석 중...")
    result = industry_search_agent(state)
    print("✅ [INDUSTRY_SEARCH] 완료 - 산업 분석 데이터 수집 완료")
    return result

def logged_market_eval(state: State) -> State:
    company = state.get("current_company", "Unknown")
    print(f"\n📈 [MARKET_EVAL] 시작 - {company} 시장성 평가 중...")
    result = market_eval_agent(state)
    print(f"✅ [MARKET_EVAL] 완료 - {company} 시장성 분석 완료")
    return result

def logged_competitor_analysis(state: State) -> State:
    company = state.get("current_company", "Unknown")
    print(f"\n🔍 [COMPETITOR_ANALYSIS] 시작 - {company} 경쟁사 분석 중...")
    result = competitor_analysis_agent(state)
    print(f"✅ [COMPETITOR_ANALYSIS] 완료 - {company} 경쟁사 분석 완료")
    return result

def logged_investment_decision(state: State) -> State:
    company = state.get("current_company", "Unknown")
    print(f"\n💰 [INVESTMENT_DECISION] 시작 - {company} 투자 결정 중...")
    result = investment_decision_agent(state)
    decision = result.get("investment_decision", False)
    decision_text = "투자 승인" if decision else "투자 거부"
    print(f"✅ [INVESTMENT_DECISION] 완료 - {company}: {decision_text}")
    return result

def logged_report_writer(state: State) -> State:
    company = state.get("current_company", "Unknown")
    print(f"\n📝 [REPORT_WRITER] 시작 - {company} 보고서 작성 중...")
    result = report_writer_agent(state)
    print(f"✅ [REPORT_WRITER] 완료 - {company} 보고서 작성 완료")
    return result


# === Resume Analysis Node ===
def resume_analysis_node(state: State) -> State:
    """
    기업 리스트와 보고서 작성 여부를 기반으로 상태만 갱신하고 State를 반환
    (분기 결정은 route_resume_analysis에서 수행)
    """
    remaining = len(state.get("selected_companies", []))
    print(f"\n📋 [RESUME_ANALYSIS] 시작 - 남은 기업: {remaining}개")
    
    if state.get("selected_companies"):
        # 1) 다음 기업 하나 꺼내기 (dict/list 방어)
        current_item = state["selected_companies"].pop(0)
        if isinstance(current_item, dict):
            current = (current_item.get("title") or "").strip()
        else:
            current = str(current_item or "").strip()

        state["current_company"] = current
        print(f"📋 [RESUME_ANALYSIS] 현재 분석 대상: {current}")

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
                    print(f"📋 [RESUME_ANALYSIS] {current} 태그: {tags}")
            except Exception as e:
                # 검색 실패해도 파이프라인 계속
                print(f"📋 [RESUME_ANALYSIS] VDB lookup failed for {current}: {e}")

        state["current_tags"] = tags
    else:
        print("📋 [RESUME_ANALYSIS] 분석할 기업 없음")
    
    print(f"✅ [RESUME_ANALYSIS] 완료")
    # state만 반환 (여기서 분기 라벨을 반환하면 안 됨)
    return state


def route_resume_analysis(state: State) -> str:
    """
    상태를 기준으로 다음 단계 라벨만 반환
    """
    if state.get("current_company"):
        # 방금 설정된 회사가 있으니 평가로 진행
        print("🔀 [ROUTE] resume_analysis → market_eval")
        return "market_eval"
    # 더 이상 분석할 기업이 없을 때
    if state.get("report_written"):
        print("🔀 [ROUTE] resume_analysis → end (모든 작업 완료)")
        return "end"
    print("🔀 [ROUTE] resume_analysis → report_writer (보고서 작성 필요)")
    return "report_writer"


# === Investment Decision Routing ===
def route_investment_decision(state: State) -> str:
    decision = state.get("investment_decision")
    if decision:
        print("🔀 [ROUTE] investment_decision → report_writer (투자 승인)")
        return "report_writer"
    else:
        print("🔀 [ROUTE] investment_decision → resume_analysis (투자 거부, 다음 기업)")
        return "resume_analysis"


# === Workflow 정의 ===
workflow = StateGraph(State)

# === 노드 등록 ===
workflow.add_node("startup_search", logged_startup_search)
workflow.add_node("industry_search", logged_industry_search)
workflow.add_node("resume_analysis", resume_analysis_node)   # ← 노드는 State 반환
workflow.add_node("market_eval", logged_market_eval)
workflow.add_node("competitor_analysis", logged_competitor_analysis)
workflow.add_node("investment_decision", logged_investment_decision)
workflow.add_node("report_writer", logged_report_writer)

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

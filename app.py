from state import State
from graph import investment_app

if __name__ == "__main__":
    initial_state: State = {
        "selected_companies": [],
        "current_company": None,
        "current_tags": [],
        "market_analysis": None,
        "competitor_analysis": None,
        "report_written": False,
        "investment_decision": None,
    }

    result = investment_app.invoke(initial_state)
    print("✅ 최종 실행 결과:", result)

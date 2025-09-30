# test_market_eval.py
from agents.market_eval_agent import market_eval_agent, GraphState

if __name__ == "__main__":
    # GraphState에 맞는 더미 데이터 정의
    dummy_state: GraphState = {
        "selected_companies": ["테스트컴퍼니", "알파테크", "베타모빌리티"],  # 아직 분석 대기 중인 기업들
        "current_company": "테스트컴퍼니",  # 지금 분석할 기업
        "current_tags": ["모빌리티", "구독"],  # 이 기업의 태그
        "market_analysis": "국내 모빌리티 시장은 빠르게 성장 중이며, 구독 서비스 확장이 활발하다.",  # 기존 시장 분석 메모
        "competitor_analysis": "대형 플랫폼 기업 3곳이 경쟁 중이며, 차별화 포인트는 구독 모델과 충성도 높은 고객 기반이다.",  # 경쟁사 분석
        "report_written": False,  # 아직 보고서 작성 안 됨
        "investment_decision": None,  # 투자 여부 판단은 아직 안 함
    }

    # 에이전트 실행
    new_state = market_eval_agent(dummy_state)

    # 결과 출력
    print("=== 실행 결과 ===")
    print(f"기업명: {new_state['current_company']}")
    print(f"시장성 분석:\n{new_state['market_analysis']}")

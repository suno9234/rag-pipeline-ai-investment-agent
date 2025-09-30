from typing import List, Optional
from typing_extensions import TypedDict, Annotated

class State(TypedDict):
    # 기업 탐색 관련
    selected_companies: Annotated[List[str], "분석해야 할 기업 리스트 (처리 완료 시 pop)"]
    current_company: Annotated[Optional[str], "현재 분석 중인 기업명"]
    current_tags: Annotated[List[str], "현재 기업의 태깅 목록"]

    # 분석 결과
    market_analysis: Annotated[Optional[str], "시장성 조사 결과"]
    competitor_analysis: Annotated[Optional[str], "경쟁사 분석 결과"]

    # 최종 판단 및 보고서
    report_written: Annotated[bool, "10개 기업 중 하나라도 보고서 작성 여부"]
    investment_decision: Annotated[Optional[bool], "투자 여부 판단 결과 (True/False)"]

    #첫 시작 임시 state
    input_text: Annotated[str, "ex)NextUnicorn에서 스타트업 2개 알려줘"]
    headless : Annotated[bool,"playwright headless 옵션"]
    emit_raw:Annotated[bool,"임시 출력"]

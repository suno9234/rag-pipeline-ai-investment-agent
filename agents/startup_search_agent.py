# agents/startup_agent_node.py
from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional
import json
import re

from tools.nextunicorn import nextunicorn_list_and_details_sync


class State(TypedDict, total=False):
    """
    공용 상태 타입 (LangGraph state로 그대로 사용 가능)
    - 이 노드는 모든 일을 '한 번에' 처리한다:
      1) limit 파싱 (input_text에서 'N개' 추출, 미지정시 기본값 사용)
      2) 리스트→디테일 크롤링 (단일 합성 툴)
      3) (옵션) 원본 JSON 출력
    """
    input_text: str          # 사용자가 입력한 문장 (예: "2개 알려줘")
    limit: int               # 가져올 개수. 없으면 input_text에서 파싱
    headless: bool           # 브라우저 headless 모드 (기본 True)
    emit_raw: bool           # True면 콘솔에 JSON 출력
    items: List[Dict[str, Any]]
    details: List[Dict[str, Any]]
    errors: List[str]


def startup_agent(state: State) -> State:
    """
    단일 '에이전트 노드' 함수.
    - 입력: State (input_text/limit/headless/emit_raw 중 일부 또는 전부)
    - 처리: limit 확정 → 합성 크롤링 호출 → 결과/에러 state에 채움
    - 출력: State (items, details, errors 포함)
    """
    # 1) limit 결정
    limit: int = state.get("limit", 0) or 0
    if limit <= 0:
        m = re.search(r"(\d+)\s*개", (state.get("input_text") or ""))
        limit = int(m.group(1)) if m else 2
        if limit < 1:
            limit = 1

    headless: bool = state.get("headless", True)
    emit_raw: bool = state.get("emit_raw", False)

    # 2) 크롤링 (리스트→디테일 한 번에)
    items: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    errors: List[str] = list(state.get("errors", []))

    try:
        res = nextunicorn_list_and_details_sync(limit=limit, headless=headless)
        items = res.get("items", [])
        details = res.get("details", [])
    except Exception as e:
        errors.append(str(e))

    # 3) (옵션) 원본 출력
    if emit_raw:
        payload = {"items": items, "details": details, "errors": errors}
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    # 4) 결과 병합하여 반환
    return {
        "limit": limit,
        "headless": headless,
        "emit_raw": emit_raw,
        "items": items,
        "details": details,
        "errors": errors,
    }


# --- 단독 실행 테스트 ---
if __name__ == "__main__":
    s: State = {
        "input_text": "NextUnicorn에서 스타트업 2개 알려줘",
        "headless": True,
        "emit_raw": True,  # 콘솔에 JSON 원본 출력
    }
    s = startup_agent(s)

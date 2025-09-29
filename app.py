# agent/graph.py
from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import asyncio

from tools.websearch import web_search

# ---- State 정의
class SearchState(TypedDict, total=False):
    query: str
    search_params: dict
    search_results: dict

# ---- Tool Wrappers (LangGraph ToolNode는 sync 함수를 기대하는 경우가 많아서 래퍼 제공)
def run_web_search_sync(state: SearchState) -> dict:
    query = state.get("query", "")
    params = state.get("search_params", {}) or {}
    # 기본값
    params.setdefault("max_results", 8)
    params.setdefault("visit_top_k", 2)
    params.setdefault("headless", True)
    # async -> sync 실행
    res = asyncio.run(web_search(query, **params))
    return {"search_results": res}

web_search_node = ToolNode("web_search", func=run_web_search_sync)

# ---- 그래프 구성: Start -> web_search_node -> END
graph = StateGraph(SearchState)
graph.add_node(web_search_node)
graph.set_entry_point(web_search_node)
graph.set_finish_point(web_search_node)

# 사용 예시(실행은 나중에)
# state = {"query": "한국 스타트업 투자 동향 2025 site:news"}
# out = graph.run(state)

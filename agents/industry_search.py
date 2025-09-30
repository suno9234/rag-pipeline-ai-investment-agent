from typing import Dict, Any
from pathlib import Path
from tools.industry_search_tool import run_search
from tools.industry_embedding_tool import industry_embedding
from langgraph.graph import StateGraph


def industry_search_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    
    industry = state.get("industry", "모빌리티")
    groups = state.get("groups", ["전기차", "전동킥보드", "자율주행"])
    use_global_sources = state.get("use_global_sources", False)

    # === 1. 검색 ===
    out_json_path = Path("docs") / f"{industry}_search_results.json"
    search_json = run_search(industry, groups, out_json_path, use_global_sources)

    # === 2. 임베딩 ===
    industry_embedding(search_json)  
    return state




# # === State 정의 ===
# from typing import TypedDict, List



# class IndustryState(TypedDict):
#     industry: str
#     groups: List[str]
#     use_global_sources: bool
#     search_json_path: str
#     embedding_done: bool

# def main():
#     # === 그래프 초기화 ===
#     graph = StateGraph(IndustryState)
#     graph.add_node("industry_pipeline", industry_pipeline_node)
#     graph.set_entry_point("industry_pipeline")
#     graph.set_finish_point("industry_pipeline")

#     app = graph.compile()

#     # === 테스트 실행 ===
#     init_state: IndustryState = {
#         "industry": "모빌리티",
#         "groups": ["전기차", "전동킥보드", "자율주행"],
#         "use_global_sources": False,
#         "search_json_path": "",
#         "embedding_done": False,
#     }

#     result = app.invoke(init_state)
#     print("\n=== 최종 결과 ===")
#     print(result)

# if __name__ == "__main__":
#     main()


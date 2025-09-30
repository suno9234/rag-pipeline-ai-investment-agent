# app.py
from langgraph.graph import StateGraph, END
from agents.startup_search_agent import State, startup_agent

def build_graph():
    g = StateGraph(State)
    g.add_node("startup_agent", startup_agent)  # 노드 1개만 등록
    g.set_entry_point("startup_agent")
    g.add_edge("startup_agent", END)
    return g.compile()

if __name__ == "__main__":
    app = build_graph()
    app.invoke({
        "input_text": "NextUnicorn에서 스타트업 2개 알려줘",
        "headless": True,
        "emit_raw": True,  # 콘솔에 원본 JSON 출력
    })

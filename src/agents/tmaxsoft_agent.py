from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_config
from typing import TypedDict, List
from src.agents import common
from src.agents.chatbot_graph import chatbot as chatbot_agent
from .probject_graph import probject_chatbot as proobject_agent
from langchain_community.tools import DuckDuckGoSearchResults
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

class AgentState(TypedDict):
    query: str
    context: list[Document]
    answer: str

class Route(BaseModel):
    target: Literal["web_search", "proobject_agent", "chatbot"] = Field(
        description="The target agent to route to")

router_system_prompt = """
Route the user's question to one of: 'proobject_agent', 'chatbot', 'web_search'.
Use 'proobject_agent' for tmaxsoft ProObject framework questions.
Use 'chatbot' if the question can be answered directly by a general LLM without web search.
Use 'web_search' only if external or up-to-date information is required.
"""

router_prompt = ChatPromptTemplate.from_messages([
    ('system', router_system_prompt),
    ('user', '{query}')
])

router_llm = common.get_model()
structured_router_llm = router_llm.with_structured_output(Route)

def router(state: AgentState) -> Literal["web_search", "proobject_agent", "chatbot"]:
    query = state["query"]
    router_chain = router_prompt | structured_router_llm
    route = router_chain.invoke({"query": query})
    return route.target

def web_search(state: AgentState) -> AgentState:
    print(f"web_search: {state['query']}")
    tool = DuckDuckGoSearchResults(output_format="list")

    result = tool.invoke(state["query"])
    result = result[0] if type(result) == list else None
    if result is None:
        return {**state,
                "answer": "검색 결과를 찾을 수 없습니다."}

    answer = f"{result['snippet']}\n\n출처: {result['link']}"

    return {**state,
            "answer": answer}

async def chatbot_node_wrapper(state: AgentState) -> AgentState:
    """
    chatbot 에이전트를 호출하는 래퍼 함수.
    service에서 ainvoke(..., config=...)로 넘긴 config는 노드 실행 시
    runnable context에 있으므로 get_config()로 가져옴.
    """
    # 노드가 그래프 안에서 실행 중일 때만 사용 가능 (service에서 준 config와 동일)
    try:
        config = get_config()
        thread_id = config.get("configurable", {}).get("thread_id")
    except RuntimeError:
        thread_id = None

    invoke_config = {"configurable": {"thread_id": thread_id}} if thread_id else None

    result = await chatbot_agent.ainvoke(
        {"query": [HumanMessage(content=state["query"])]},
        config=invoke_config,
    )
    return {**state, "answer": result["answer"]}

graph_builder = StateGraph(AgentState)
graph_builder.add_node("web_search", web_search)
graph_builder.add_node("proobject_agent", proobject_agent)
graph_builder.add_node("chatbot", chatbot_node_wrapper)
graph_builder.add_conditional_edges(
    START,
    router,
    {
        "web_search": "web_search",
        "proobject_agent": "proobject_agent",
        "chatbot": "chatbot",
    }
)

from .common import checkpointer

graph_builder.add_edge("web_search", END)
graph_builder.add_edge("proobject_agent", END)
graph_builder.add_edge("chatbot", END)
tmaxsoft_agent = graph_builder.compile(checkpointer=checkpointer)
from .chatbot_graph import chatbot
from .probject_graph import probject_chatbot as proobject_agent
from .tmaxsoft_agent import tmaxsoft_agent
from .mcp_client_agent import mcp_agent
from .lazy_loading_agent import LazyLoadingAgent
from dataclasses import dataclass
from langgraph.pregel import Pregel
from typing import cast

@dataclass
class Agent:
    """에이전트 메타데이터 및 그래프"""
    description: str
    graph_like: Pregel | LazyLoadingAgent       
    
agents: dict[str, Agent] = {
    "chatbot": Agent("this is simple chatbot", chatbot),
    "probject_chatbot": Agent("this is RAG agent", proobject_agent),
    "tmaxsoft_agent": Agent("this is tmaxsoft agent", tmaxsoft_agent),
    "mcp_client_agent": Agent("this is mcp client agent", mcp_agent),
}

async def get_lazy_agent(agent_key: str) -> LazyLoadingAgent | None:
    """
    LazyLoadingAgent인 경우 첫 요청 시점에 load() 후 그래프를 반환
    """
    agent_wrapper = agents[agent_key]
    graph_like = agent_wrapper.graph_like
    if isinstance(graph_like, LazyLoadingAgent):
        lazy_agent = cast(LazyLoadingAgent, graph_like)
        if not lazy_agent._loaded:
            await lazy_agent.load()
        return lazy_agent
    return graph_like

from .chatbot_graph import chatbot
from .probject_graph import probject_chatbot as proobject_agent
from .tmaxsoft_agent import tmaxsoft_agent
from dataclasses import dataclass
from langgraph.pregel import Pregel

@dataclass
class Agent:
    """에이전트 메타데이터 및 그래프"""
    description: str
    graph_like: Pregel


agents: dict[str, Agent] = {
    "chatbot": Agent("this is simple chatbot", chatbot),
    "probject_chatbot": Agent("this is RAG agent", proobject_agent),
    "tmaxsoft_agent": Agent("this is tmaxsoft agent", tmaxsoft_agent),
}


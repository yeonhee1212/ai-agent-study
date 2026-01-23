from .chatbot_agent import chatbot
from .probject_chatbot import probject_chatbot
from dataclasses import dataclass
from langgraph.pregel import Pregel

@dataclass
class Agent:
    """에이전트 메타데이터 및 그래프"""
    description: str
    graph_like: Pregel


agents: dict[str, Agent] = {
    "chatbot": Agent("this is simple chatbot", chatbot),
    "probject_chatbot": Agent("this is RAG agent", probject_chatbot),
}


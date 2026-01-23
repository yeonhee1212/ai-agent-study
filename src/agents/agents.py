from .chatbot_agent import chatbot
from .self_rag_agent import self_rag_agent
from dataclasses import dataclass
from langgraph.pregel import Pregel

@dataclass
class Agent:
    description: str
    graph_like: Pregel

agents : dict[str, Agent] = {
    "chatbot" : Agent("this is simple chatbot", chatbot),
    "self_rag_agent": Agent("this is RAG agent", self_rag_agent)
}


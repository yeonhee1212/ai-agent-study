from .chatbot import chatbot
from dataclasses import dataclass
from langgraph.pregel import Pregel

@dataclass
class Agent:
    description: str
    graph_like: Pregel

agents : dict[str, Agent] = {
    "chatbot" : Agent("this is simple chatbot", chatbot)
}


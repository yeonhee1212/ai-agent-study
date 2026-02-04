from abc import ABC, abstractmethod
from langgraph.pregel import Pregel
from langgraph.graph import StateGraph

class LazyLoadingAgent(ABC):
    def __init__(self):
        self._name = None
        self._loaded = False
        self._graph: StateGraph | Pregel | None = None

    def get_graph(self) -> StateGraph | Pregel:
        if not self._loaded:
            raise RuntimeError(f"Agent {self._name} is not loaded")
        if self._graph is None:
            raise RuntimeError(f"Agent {self._name} graph is not loaded")
        return self._graph

    @abstractmethod
    async def load(self) -> None:
        raise NotImplementedError
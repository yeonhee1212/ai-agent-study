from .llm_model import get_model
from langchain_mcp_adapters.client import MultiServerMCPClient, StreamableHttpConnection
from langchain.agents import create_agent
import asyncio
import src.config as config
from .lazy_loading_agent import LazyLoadingAgent
from langgraph.graph.state import CompiledStateGraph

mcp_system_prompt = """
당신은 MCP 도구를 사용해 사용자 질문에 답하는 에이전트입니다.

## 사용 가능한 도구
1. **retrieve**: 벡터 스토어에서 질문과 관련된 문서를 검색합니다. 질문 문자열을 넘기면 문서 목록(page_content, metadata)을 반환합니다.
2. **check_relevance**: 질문과 검색된 문서 목록의 관련성을 0~1 점수로 반환합니다. query 와 document(검색 결과 리스트)를 넘깁니다.
3. **web_search**: 웹 검색을 수행해 스니펫과 출처 링크를 반환합니다. 질문 또는 검색어를 넘깁니다.

## 답변 절차 (반드시 이 순서를 따르세요)
1. 사용자 질문이 들어오면 먼저 **retrieve**를 호출해 관련 문서를 검색하세요. 예: retrieve(human_message="사용자 질문")
2. **retrieve의 반환값을 변수에 저장**한 뒤, **check_relevance**를 호출할 때:
- `query` 파라미터: 사용자의 원래 질문 문자열
- `document` 파라미터: retrieve가 반환한 검색 결과 리스트를 그대로 전달
예: check_relevance(query="사용자 질문", document=retrieve_결과)
3. **check_relevance**의 반환값인 관련성 점수를 확인하세요:
   - 점수 >= 0.6: web_search 호출하지 않고 검색된 문서만 사용
   - 점수 < 0.6: web_search 호출하여 웹 검색 결과도 함께 사용
5. 답변 시 문서의 슬라이드 번호나 **web_search**시에 참고한 링크가 있으면 명시하세요.
"""

# MCP 서버 연결: config의 MCP_SERVER_HOST, MCP_SERVER_PORT 사용 (streamable-http 기본 경로 /mcp)
MCP_SERVER_URL = f"http://{config.MCP_SERVER_HOST}:{config.MCP_SERVER_PORT}/mcp"

class MCPClientAgent(LazyLoadingAgent):
    def __init__(self) -> None:
        super().__init__()
        self._name = "mcp_client_agent"
        self._mcp_tools = None
        self._mcp_client = None

    async def load(self) -> None:
        try:
            self._mcp_client = MultiServerMCPClient(
                {
                    "local_mcp_server": StreamableHttpConnection(
                        transport="streamable_http",
                        url=MCP_SERVER_URL,
                        headers={
                            "Content-Type": "application/json",
                        },
                    )
                }
            )
            self._mcp_client = MultiServerMCPClient(
                {
                    "local_mcp_server": StreamableHttpConnection(
                        transport="streamable_http",
                        url=MCP_SERVER_URL,
                        headers={
                            "Content-Type": "application/json",
                        },
                    )
                }
            )
            self._mcp_tools = await self._mcp_client.get_tools()
            self._graph = self._create_graph()
            self._loaded = True
        except Exception as e:
            self._mcp_tools = []
            self._mcp_client = None
            raise e

    def _create_graph(self) -> CompiledStateGraph:
        return create_agent(
                    model=get_model(),
                    tools=self._mcp_tools,
                    name=self._name,
                    system_prompt=mcp_system_prompt
                )

mcp_agent = MCPClientAgent()
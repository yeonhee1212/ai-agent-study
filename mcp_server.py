from langchain.agents.structured_output import StructuredOutputError
from langchain_core.messages import AIMessage
from mcp.server.fastmcp import FastMCP
from src.agents.common import llm_model
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE, RAG_COLLECTION_NAME, RAG_PERSIST_DIRECTORY, RAG_RETRIEVER_K
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_classic import hub
from langchain_community.tools import DuckDuckGoSearchResults
import src.config as config
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

mcp = FastMCP(
    name="mcp-server",
    instructions="You are a helpful assistant that can answer questions and help with tasks.",
    host=config.MCP_SERVER_HOST,
    port=int(config.MCP_SERVER_PORT),
)

_embedding_model = None
_vector_store = None
_retriever = None

def _get_retriever():
    """Retriever를 지연 로딩으로 초기화"""
    global _embedding_model, _vector_store, _retriever
    
    if _retriever is None:
        print("[ProObject Agent] 임베딩 모델 및 벡터 스토어 초기화 중...")
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        
        _vector_store = Chroma(
            collection_name=RAG_COLLECTION_NAME,
            persist_directory=RAG_PERSIST_DIRECTORY,
            embedding_function=_embedding_model,
        )
        
        _retriever = _vector_store.as_retriever(search_kwargs={"k": RAG_RETRIEVER_K})
        print("[ProObject Agent] 초기화 완료")
    
    return _retriever

@mcp.tool(name="retrieve", description="벡터 스토어에서 관련 문서를 검색")
def retrieve(human_message: str) -> list[dict]:
    retriever = _get_retriever()
    document_result_list: list[Document] = retriever.invoke(human_message)
    result_list: list[dict] = []
    for document in document_result_list:
        result_list.append({
            "page_content": document.page_content,
            "metadata": document.metadata,
        })
    return result_list

@mcp.tool(
    name="check_relevance",
    description="질문과 문서 목록의 관련성을 0~1 점수로 반환합니다. query(질문 문자열)와 document(retrieve 도구의 반환값, 검색 결과 리스트) 두 파라미터를 모두 필수로 전달해야 합니다. document는 retrieve 도구를 먼저 호출하여 얻은 결과를 그대로 전달하세요."
)
def check_relevance(query: str, document: list[dict]) -> float:
    doc_relevance_prompt = PromptTemplate.from_template(""" You are an expert retrieval evaluator.
        Evaluate how relevant the given documents are to the user's question.
        Scoring rules:
        - 1.0: documents directly and fully answer the question
        - 0.8: strongly relevant but partially incomplete
        - 0.6: somewhat relevant
        - 0.2 ~ 0.5: weakly relevant
        - 0.0: completely irrelevant
        Return ONLY a float number between 0.0 and 1.0.
        Do not explain.

        Question:
        {query}

        Documents:
        {document}
            """)
    doc_relevance_chain = doc_relevance_prompt | llm_model | StrOutputParser()
    response :str = doc_relevance_chain.invoke({
        "query": query,
        "document": document,
        })

    return float(response)

@mcp.tool(name="web_search", description="웹 검색을 수행하고 결과를 반환")
def web_search(query: str) -> str:
    tool = DuckDuckGoSearchResults(output_format="list")

    result = tool.invoke(query)
    result = result[0] if type(result) == list else None
    if result is None:
        return "검색 결과를 찾을 수 없습니다."

    answer = f"{result['snippet']}\n\n출처: {result['link']}"

    return answer

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
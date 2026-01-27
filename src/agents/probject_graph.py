from src.config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    RAG_COLLECTION_NAME,
    RAG_PERSIST_DIRECTORY,
    RAG_RETRIEVER_K,
)
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
from langchain_chroma import Chroma
from .common import llm_model, checkpointer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

class AgentState(TypedDict):
    query: str
    context: List[Document]
    answer: str


# 임베딩 모델 및 벡터 스토어 초기화
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": EMBEDDING_DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)

vector_store = Chroma(
    collection_name=RAG_COLLECTION_NAME,
    persist_directory=RAG_PERSIST_DIRECTORY,
    embedding_function=embedding_model,
)

retriever = vector_store.as_retriever(search_kwargs={"k": RAG_RETRIEVER_K})

def retrieve(state: AgentState) -> AgentState:
    """
    벡터 스토어에서 관련 문서를 검색
    """
    print(f"retrieve: {state['query']}")
    query = state["query"]
    docs = retriever.invoke(query)
    print(f"docs: {docs}")
    return AgentState(query=query, context=docs, answer="")

def generate_answer(state: AgentState) -> AgentState:
    """
    RAG 컴포넌트에서 답변을 생성
    """
    query = state["query"]
    context = state["context"]

    system_prompt = (
    "당신은 tmaxsoft의 proobject제품의 엔지니어입니다. "
    "context를 참고하여 context에 있는 내용을 바탕으로 답변을 생성하세요. "
    "답변을 생성할 때 슬라이드의 몇번째 페이지를 참고하여 답변했는지 슬라이드 번호를 알려주세요."
)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template( "context: {context}\n\nquery: {query}"),
    ])

    reg_chain = prompt | llm_model

    # context를 문자열로 변환
    context_str = "\n\n".join([doc.page_content for doc in context])
    
    response = reg_chain.invoke({
        "query": query,
        "context": context_str,
    })

    return AgentState(query=query, context=context, answer=response.content)

# 그래프 정의 및 컴파일
def create_probject_chatbot():
    """ProObject 챗봇 그래프 생성"""
    graph_builder = StateGraph(AgentState)
    
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate_answer", generate_answer)
    
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate_answer")
    graph_builder.add_edge("generate_answer", END)
    
    return graph_builder.compile(checkpointer=checkpointer)


probject_chatbot = create_probject_chatbot()

from typing import Optional
from langchain_core.messages import BaseMessage
from langgraph.func import entrypoint
from .common import llm_model, checkpointer
from src.config import CHATBOT_MAX_HISTORY

@entrypoint(checkpointer=checkpointer)
async def chatbot(
    inputs: dict[str, list[BaseMessage]],
    *,
    previous: Optional[dict[str, list[BaseMessage]]] = None,
) -> dict:
    """
    LangGraph 기반의 간단한 채팅봇 에이전트
    
    Args:
        inputs: 입력 메시지 딕셔너리 ({"query": list[BaseMessage]})
        previous: 이전 대화 상태 (선택사항)
    
    Returns:
        답변을 포함한 최종 상태
    """
    input_query = inputs["query"]

    # 이전 대화가 있으면 합치기
    if previous:
        all_messages = previous["query"] + input_query
    else:
        all_messages = input_query

    # 최근 대화만 유지 (새 입력 메시지 포함)
    if len(all_messages) > CHATBOT_MAX_HISTORY:
        all_messages = all_messages[-CHATBOT_MAX_HISTORY:]

    # 이전 대화 맥락을 포함한 메시지 리스트로 LLM 호출
    response = await llm_model.ainvoke(all_messages)
    
    # 저장할 때도 최근 대화만 유지 (새 응답 포함)
    all_messages = all_messages + [response]
    if len(all_messages) > CHATBOT_MAX_HISTORY:
        all_messages = all_messages[-CHATBOT_MAX_HISTORY:]
    
    print(f"[저장] 대화 히스토리 {len(all_messages)}개 저장 (최대 {CHATBOT_MAX_HISTORY}개)")
    
    return entrypoint.final(
        value={"answer": response.content},
        save={"query": all_messages},
    )
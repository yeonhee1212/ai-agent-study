from typing import Optional
from langchain_core.messages import BaseMessage
from langgraph.func import entrypoint
from .common import llm_model, checkpointer

## Langgraph 기반의 간단한 채팅봇 에이전트
@entrypoint(checkpointer=checkpointer)
async def chatbot(
    inputs: dict[str, list[BaseMessage]],
    *,
    previous: Optional[dict[str, list[BaseMessage]]] = None
): 
    input_query = inputs["query"] # 입력 메시지 추출

    if previous:
        messages = previous["query"] + input_query

    # 최근 대화 5개만 유지 (새 입력 메시지 포함)
    MAX_HISTORY = 5
    if len(input_query) > MAX_HISTORY:
        input_query = input_query[-MAX_HISTORY:]

    response = await llm_model.ainvoke(input_query)
    
    # 저장할 때도 최근 5개만 유지 (새 응답 포함)
    all_messages = input_query + [response]
    if len(all_messages) > MAX_HISTORY:
        all_messages = all_messages[-MAX_HISTORY:]
    
    print(f"[저장] 대화 히스토리 {len(all_messages)}개 저장 (최대 {MAX_HISTORY}개)")
    
    return entrypoint.final(
        value={"answer": response.content}, save={"query": all_messages}
    )
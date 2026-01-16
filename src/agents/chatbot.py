from typing import Optional
from langchain_core.messages import BaseMessage
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import MemorySaver
from .llm_model import get_model

# 모델 인스턴스를 모듈 레벨에서 한 번만 생성
model = get_model("GPT_4_MINI")
print(f"[초기화] 연결된 llm model : {model.model_name}")

# 체크포인터를 생성하여 상태 저장/복원
checkpointer = MemorySaver()

## Langgraph 기반의 간단한 채팅봇 에이전트
@entrypoint(checkpointer=checkpointer)
async def chatbot(
    inputs: dict[str, list[BaseMessage]],
    *,
    previous: Optional[dict[str, list[BaseMessage]]] = None
): 
    messages = inputs["messages"] # 입력 메시지 추출

    if previous:
        messages = previous["messages"] + messages

    # 최근 대화 5개만 유지 (새 입력 메시지 포함)
    MAX_HISTORY = 5
    if len(messages) > MAX_HISTORY:
        messages = messages[-MAX_HISTORY:]

    response = await model.ainvoke(messages)
    
    # 저장할 때도 최근 5개만 유지 (새 응답 포함)
    all_messages = messages + [response]
    if len(all_messages) > MAX_HISTORY:
        all_messages = all_messages[-MAX_HISTORY:]
    
    print(f"[저장] 대화 히스토리 {len(all_messages)}개 저장 (최대 {MAX_HISTORY}개)")
    
    return entrypoint.final(
        value={"messages": [response]}, save={"messages": all_messages}
    )
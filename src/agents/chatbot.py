from langchain_core.messages import BaseMessage
from langgraph.func import entrypoint
from .llm_model import get_model

## Langgraph 기반의 간단한 채팅봇 에이전트
@entrypoint()
async def chatbot(
    inputs: dict[str, list[BaseMessage]],
    *,
    previous: dict[str,list[BaseMessage]]
): 
    messages = inputs["messages"] # 입력 메시지 추출

    if previous:
        messages = previous["messages"] + messages

    model = get_model()
    print(f"연결된 llm model : {model.model}")
    response = await model.ainvoke(messages)
    return entrypoint.final(
        value={"messages": [response]}, save={"messages": messages + [response]}
    )
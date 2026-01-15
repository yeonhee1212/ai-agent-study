
# from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.func import entrypoint
## Langgraph 기반의 간단한 채팅봇 에이전트

@entrypoint()
async def chatbot(
    inputs: dict[str, list[BaseMessage]],
    *,
    previous: dict[str,list[BaseMessage]],
    config: RunnableConfig,
): 
    messages = inputs["messages"] # 입력 메시지 추출

    if previous:
        messages = previous["messages"] + messages

    # model = ChatOllama(model=settings.OLLAMA_MODEL, temperature=0.5)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    response = await model.ainvoke(messages)
    return entrypoint.final(
        value={"messages": [response]}, save={"messages": messages + [response]}
    )
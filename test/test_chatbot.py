# test_chatbot.py
import sys
sys.path.append(".")
import asyncio
from langchain_core.messages import HumanMessage
from src.agents.agents import agents

async def main():

    while True:
        message = input("명령어를 입력하세요 (exit 입력 시 종료): ")
        if message == "exit":
            break

        result = await agents["chatbot"].graph_like.ainvoke(
            {
                "messages": [HumanMessage(content=message)]
            }
        )

    # LLM 응답 출력
        ai_message = result["messages"][-1]
        print(f"AI: {ai_message.content}")


asyncio.run(main())

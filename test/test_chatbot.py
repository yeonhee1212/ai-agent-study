# test_chatbot.py
import sys
import asyncio
import os
from dotenv import load_dotenv
sys.path.append(".")

from langchain_core.messages import HumanMessage
from src.agents.chatbot import chatbot

load_dotenv()

async def main():
    result = await chatbot.ainvoke(
        {
            "messages": [HumanMessage(content="안녕")]
        },
        config={
            "configurable": {
                "model": "gpt-4o-mini"
            }
        }
    )

    print(result)

asyncio.run(main())

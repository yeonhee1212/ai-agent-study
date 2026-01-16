from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from src.agents.agents import agents
import sys
sys.path.append(".")
from src.schema.schema import ChatRequest, ChatResponse, Message

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/chatbot", response_model=ChatResponse)
async def chatbot(request: ChatRequest):
    agent = agents["chatbot"]

    humaman_messages = [
        HumanMessage(content=msg.content) for msg in request.messages if msg.role == "user"
    ]

    result = await agent.graph_like.ainvoke(
        {
            "messages": humaman_messages
        },
        config={"configurable": {"thread_id": request.thread_id or "default"}}
    )

    ai_response = result["messages"][-1]

    return ChatResponse(message=Message(role="assistant", content=ai_response.content))
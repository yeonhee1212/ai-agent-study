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

    # chatbot_agent는 {"query": list[BaseMessage]} 형식을 기대
    human_messages = [
        HumanMessage(content=msg.content) for msg in request.messages if msg.role == "user"
    ]

    result = await agent.graph_like.ainvoke(
        {
            "query": human_messages
        },
        config={"configurable": {"thread_id": request.thread_id or "default"}}
    )

    ai_response_content = result["answer"]

    return ChatResponse(message=Message(role="assistant", content=ai_response_content))


@app.post("/self_rag_agent", response_model=ChatResponse)
async def self_rag_agent(request: ChatRequest):
    agent = agents["self_rag_agent"]

    # self_rag_agent는 AgentState (query: str) 형식을 기대
    # 마지막 사용자 메시지의 내용을 query로 사용
    user_messages = [msg.content for msg in request.messages if msg.role == "user"]
    query = user_messages[-1] if user_messages else ""

    result = await agent.graph_like.ainvoke(
        {
            "query": query,
            "context": [],
            "answer": ""
        },
        config={"configurable": {"thread_id": request.thread_id or "default"}}
    )

    # self_rag_agent는 AgentState (answer: str) 형식으로 반환
    ai_response_text = result["answer"]

    return ChatResponse(message=Message(role="assistant", content=ai_response_text))
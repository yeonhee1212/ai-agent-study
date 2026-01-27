from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from src.agents.agents import agents
from src.schema.schema import ChatRequest, ChatResponse, Message
from src.config import DEFAULT_THREAD_ID

app = FastAPI()


def get_user_messages(request: ChatRequest) -> list[str]:
    """요청에서 사용자 메시지 내용 추출"""
    return [msg.content for msg in request.messages if msg.role == "user"]


def get_thread_id(request: ChatRequest) -> str:
    """요청에서 thread_id 추출, 없으면 기본값 반환"""
    return request.thread_id or DEFAULT_THREAD_ID


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.post("/chatbot", response_model=ChatResponse)
async def chatbot(request: ChatRequest):
    """
    일반 챗봇 엔드포인트
    chatbot_agent는 {"query": list[BaseMessage]} 형식을 기대
    """
    try:
        agent = agents["chatbot"]
        
        # 사용자 메시지를 HumanMessage로 변환
        human_messages = [
            HumanMessage(content=msg.content)
            for msg in request.messages if msg.role == "user"
        ]
        
        if not human_messages:
            raise HTTPException(status_code=400, detail="사용자 메시지가 없습니다.")

        result = await agent.graph_like.ainvoke(
            {"query": human_messages},
            config={"configurable": {"thread_id": get_thread_id(request)}},
        )

        ai_response_content = result["answer"]
        return ChatResponse(
            message=Message(role="assistant", content=ai_response_content)
        )
    except KeyError:
        raise HTTPException(status_code=500, detail="chatbot 에이전트를 찾을 수 없습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"에러 발생: {str(e)}")


@app.post("/probject_chatbot", response_model=ChatResponse)
async def probject_chatbot(request: ChatRequest):
    """
    ProObject RAG 챗봇 엔드포인트
    probject_chatbot은 AgentState (query: str) 형식을 기대
    """
    try:
        agent = agents["probject_chatbot"]
        
        # 마지막 사용자 메시지의 내용을 query로 사용
        user_messages = get_user_messages(request)
        if not user_messages:
            raise HTTPException(status_code=400, detail="사용자 메시지가 없습니다.")
        
        query = user_messages[-1]

        result = await agent.graph_like.ainvoke(
            {
                "query": query,
                "context": [],
                "answer": "",
            },
            config={"configurable": {"thread_id": get_thread_id(request)}},
        )

        ai_response_text = result["answer"]
        return ChatResponse(
            message=Message(role="assistant", content=ai_response_text)
        )
    except KeyError:
        raise HTTPException(
            status_code=500, detail="probject_chatbot 에이전트를 찾을 수 없습니다."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"에러 발생: {str(e)}")


@app.post("/tmaxsoft_agent", response_model=ChatResponse)
async def tmaxsoft_agent(request: ChatRequest):
    """
    Tmaxsoft agent 엔드포인트
    tmaxsoft_agent은 AgentState (query: str) 형식을 기대
    """
    try:
        agent = agents["tmaxsoft_agent"]
        
        # 마지막 사용자 메시지의 내용을 query로 사용
        user_messages = get_user_messages(request)
        if not user_messages:
            raise HTTPException(status_code=400, detail="사용자 메시지가 없습니다.")
        
        query = user_messages[-1]

        result = await agent.graph_like.ainvoke(
            {
                "query": query,
                "context": [],
                "answer": ""
            },
            config={"configurable": {"thread_id": get_thread_id(request)}},
        )

        ai_response_text = result["answer"]
        return ChatResponse(
            message=Message(role="assistant", content=ai_response_text)
        )
    except KeyError:
        raise HTTPException(
            status_code=500, detail="tmaxsoft_agent 에이전트를 찾을 수 없습니다."
        )
    except Exception as e:
        print(f"에러 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"에러 발생: {str(e)}")
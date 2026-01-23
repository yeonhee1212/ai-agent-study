import streamlit as st
import requests
import uuid


st.set_page_config(page_title="AI Agent Chat", layout="centered", page_icon=":robot:")
st.title("AI Agent Chat")

# 에이전트 선택 드롭다운
if "selected_agent" not in st.session_state:
    st.session_state.selected_agent = "chatbot"

selected_agent = st.selectbox(
    "에이전트 선택",
    options=["chatbot", "rag_agent"],
    index=0 if st.session_state.selected_agent == "chatbot" else 1,
    key="agent_selector"
)

# 에이전트가 변경되면 대화 히스토리 초기화
if st.session_state.selected_agent != selected_agent:
    st.session_state.selected_agent = selected_agent
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid.uuid4())
    st.rerun()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력
user_input = st.chat_input("메시지를 입력하세요")

if user_input:
    st.session_state.messages.append({
        "role" : "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    payload = {
        "messages": [
            {
                "role": "user",
                "content": user_input
            }
        ],
        "thread_id": st.session_state.thread_id
    }

    with st.chat_message("assistant"):
        with st.spinner("AI가 메시지를 생성중입니다..."):
            # 선택한 에이전트에 따라 엔드포인트 결정
            endpoint_map = {
                "chatbot": "/chatbot",
                "rag_agent": "/self_rag_agent"
            }
            endpoint = endpoint_map.get(selected_agent, "/chatbot")
            
            ai_response = requests.post(
                f"http://localhost:8000{endpoint}",
                json=payload,
                timeout=600
            )
            try:
                ai_response.raise_for_status()
                ai_response_text = ai_response.json()["message"]["content"]
                st.markdown(ai_response_text)
            except Exception as e:
                st.error("AI 응답을 받는 중 오류가 발생했습니다.")
                st.error(f"오류 메시지: {e}")
                ai_response_text = "오류가 발생했습니다. 다시 시도해주세요."


    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_response_text
    })

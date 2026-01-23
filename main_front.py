import streamlit as st
import requests
import uuid
from src.config import (
    SERVER_URL,
    FRONTEND_API_TIMEOUT,
    ENDPOINT_CHATBOT,
    ENDPOINT_PROJECT_CHATBOT,
)

# 페이지 설정
st.set_page_config(
    page_title="AI Agent Chat", layout="centered", page_icon=":robot:"
)
st.title("AI Agent Chat")

# 에이전트 옵션
AGENT_OPTIONS = ["chatbot", "probject_chatbot"]
ENDPOINT_MAP = {
    "chatbot": ENDPOINT_CHATBOT,
    "probject_chatbot": ENDPOINT_PROJECT_CHATBOT,
}

# 세션 상태 초기화
if "selected_agent" not in st.session_state:
    st.session_state.selected_agent = AGENT_OPTIONS[0]

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# 에이전트 선택 드롭다운
selected_agent = st.selectbox(
    "에이전트 선택",
    options=AGENT_OPTIONS,
    index=AGENT_OPTIONS.index(st.session_state.selected_agent),
    key="agent_selector",
)

# 에이전트가 변경되면 대화 히스토리 초기화
if st.session_state.selected_agent != selected_agent:
    st.session_state.selected_agent = selected_agent
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid.uuid4())
    st.rerun()

# 기존 대화 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력 처리
user_input = st.chat_input("메시지를 입력하세요")

if user_input:
    # 사용자 메시지 추가 및 표시
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # API 요청 준비
    payload = {
        "messages": [{"role": "user", "content": user_input}],
        "thread_id": st.session_state.thread_id,
    }
    endpoint = ENDPOINT_MAP.get(selected_agent, ENDPOINT_CHATBOT)
    url = f"{SERVER_URL}{endpoint}"

    # AI 응답 요청
    with st.chat_message("assistant"):
        with st.spinner("AI가 메시지를 생성중입니다..."):
            try:
                response = requests.post(
                    url, json=payload, timeout=FRONTEND_API_TIMEOUT
                )
                response.raise_for_status()
                ai_response_text = response.json()["message"]["content"]
                st.markdown(ai_response_text)
            except requests.exceptions.RequestException as e:
                st.error("AI 응답을 받는 중 오류가 발생했습니다.")
                st.error(f"오류 메시지: {str(e)}")
                ai_response_text = "오류가 발생했습니다. 다시 시도해주세요."
            except (KeyError, ValueError) as e:
                st.error("응답 형식이 올바르지 않습니다.")
                st.error(f"오류 메시지: {str(e)}")
                ai_response_text = "응답을 파싱하는 중 오류가 발생했습니다."

    # AI 응답을 메시지 히스토리에 추가
    st.session_state.messages.append(
        {"role": "assistant", "content": ai_response_text}
    )

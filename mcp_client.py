import streamlit as st
import uuid
import src.config as config
import requests

# 페이지 설정
st.set_page_config(
    page_title="MCP Client Agent Chat", layout="centered", page_icon=":robot:"
)
st.title("MCP Client")

# 세션 상태 초기화
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

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

    payload = {
        "messages": [{"role": "user", "content": user_input}],
        "thread_id": st.session_state.thread_id,
    }
    url = f"http://{config.MCP_SERVER_HOST}:{config.MCP_SERVER_PORT}/mcp"
    
    with st.chat_message("assistant"):
        with st.spinner("AI가 메시지를 생성중입니다..."):
            try:
                response = requests.post(url, json=payload, timeout=config.FRONTEND_API_TIMEOUT)
                response.raise_for_status()
                ai_response_text = response.json()
                st.markdown(ai_response_text)
            except requests.exceptions.RequestException as e:
                st.error("AI 응답을 받는 중 오류가 발생했습니다.")
                st.error(f"오류 메시지: {str(e)}")
                ai_response_text = "오류가 발생했습니다. 다시 시도해주세요."

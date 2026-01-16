import streamlit as st
import requests
import uuid


st.set_page_config(page_title="AI Agent Chat", layout="centered", page_icon=":robot:")
st.title("AI Agent Chat")

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
                "content":user_input
            }
        ],
        "thread_id": st.session_state.thread_id
    }

    with st.chat_message("assistant"):
        with st.spinner("AI가 메시지를 생성중입니다..."):
            ai_response = requests.post(
                "http://localhost:8000/chatbot",
                json=payload,
                timeout=60
            )
            ai_response.raise_for_status()

            ai_response_text = ai_response.json()["message"]["content"]
            st.markdown(ai_response_text)

    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_response_text
    })

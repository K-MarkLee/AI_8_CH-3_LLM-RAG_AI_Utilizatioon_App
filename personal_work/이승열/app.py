import streamlit as st
from chat_bot_test import get_response, save_to_json  # 필요한 함수 가져오기
import os
import atexit  # 애플리케이션 종료 시 실행할 함수 등록

# Streamlit 앱 제목
st.title("요리 전문가 챗봇")
st.write("질문을 입력하면 요리 관련 정보를 제공합니다.")

# 대화 히스토리 초기화 (세션 상태)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # 대화 기록 저장

if "query" not in st.session_state:
    st.session_state.query = ""  # 입력값 초기화

# **답변 표시 영역**
st.markdown("### 답변")
if "response" in st.session_state and st.session_state.response:
    st.markdown(
        f"""
        <div style="border: 2px solid #ddd; padding: 20px; height: 500px; width: 800px; overflow-y: auto; font-size: 18px; background-color: #f7f7f9; color: #333; border-radius: 8px;">
            {st.session_state.response}
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div style="border: 2px solid #ddd; padding: 20px; height: 500px; width: 800px; overflow-y: auto; font-size: 18px; background-color: #f7f7f9; color: #333; border-radius: 8px;">
            답변이 여기에 표시됩니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

# 콜백 함수 정의
def handle_input():
    query = st.session_state.query
    if query:
        # 대화 히스토리에 사용자 입력 추가
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.query = ""  # 입력 필드 초기화

        # 모델 호출
        try:
            # 히스토리를 포함한 모델 입력 구성
            conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history])
            response = get_response(conversation)  # 모델 호출에 히스토리 포함
            st.session_state.chat_history.append({"role": "assistant", "content": response.content})
            st.session_state.response = response.content  # 답변 저장
        except Exception as e:
            st.error(f"에러가 발생했습니다: {e}")

# 사용자 입력 필드 (엔터키로 제출 가능)
st.text_input(
    "질문을 입력하세요:",
    key="query",
    on_change=handle_input,  # 콜백 함수 연결
)

# 히스토리 자동 저장 기능
def auto_save_history():
    if "chat_history" in st.session_state and st.session_state.chat_history:
        save_path = "log/chat_history.json"
        try:
            save_to_json(save_path, st.session_state.chat_history)
            print(f"히스토리가 자동 저장되었습니다: {os.path.abspath(save_path)}")
        except Exception as e:
            print(f"히스토리 저장 중 에러 발생: {e}")

# 앱 종료 시 자동 저장
atexit.register(auto_save_history)

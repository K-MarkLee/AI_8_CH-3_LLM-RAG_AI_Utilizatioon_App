import streamlit as st
import logging
import os
import json
import time
from chat_bot_test import get_response, save_to_json, create_json_file  # 필요한 함수 가져오기

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 경로 설정
LOG_DIR = "log"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    


# 세션 상태 초기화
def initialize_session_state():
    """세션 상태 초기화"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
    if "messages" not in st.session_state:
        st.session_state.messages = []  # 대화 기록
    if "json_file_path" not in st.session_state:
        st.session_state.json_file_path = create_json_file(base_dir=LOG_DIR)  # JSON 파일 생성
        

# JSON에 기록 추가
def append_to_json(file_path, record):
    """JSON 파일에 기록 추가"""
    try:
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump([record], f, ensure_ascii=False, indent=4)
        else:
            with open(file_path, "r+", encoding="utf-8") as f:
                data = json.load(f)
                data.append(record)
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"JSON 파일 업데이트 중 오류 발생: {e}")
        st.error(f"JSON 파일 업데이트 중 오류 발생: {e}")


def main():
    # 페이지 설정
    st.set_page_config(
        page_title="요리 도우미",
        page_icon="🍳",
        layout="wide"
    )

    # 세션 상태 초기화
    initialize_session_state()

    st.title("요리 도우미 🍳")

    # 채팅 인터페이스
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # 사용자 입력 처리
    if query := st.chat_input("질문을 입력하세요"):
        # 사용자 메시지 저장 및 출력
        user_message = {"role": "user", "content": query}
        st.session_state.messages.append(user_message)
        with st.chat_message("user"):
            st.write(query)

        # AI 응답 생성 및 저장
        with st.chat_message("assistant"):
            with st.spinner("답변을 생성하는 중..."):
                try:
                    # AI 답변 생성
                    response = get_response(query)
                    assistant_message = {"role": "assistant", "content": response}
                    st.session_state.messages.append(assistant_message)

                    # UI에 출력
                    st.write(response.content)

                    # JSON 파일에 기록 추가
                    record = {
                        "user": query,
                        "assistant": response,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    append_to_json(st.session_state.json_file_path, record)

                except Exception as e:
                    logger.error(f"답변 생성 중 오류 발생: {e}")
                    st.error(f"답변 생성 중 오류 발생: {e}")


if __name__ == '__main__':
    main()


# 라이브러리 불러오기

import streamlit as st # 스트림릿 라이브러리
import streamlit.components.v1 as components #스트림릿에 html요소 삽입
import logging # 어플리케이션 동작 상태 기록 하는 로그
import os # 파일 경로 및 환경 변수 작업
import json # json파일 처리
import time # 타임스탬프 생성
import io # 메모리 버퍼 생성
import base64 # 데이터를 텍스트 형태로


from gtts import gTTS # 구글 TTS 라이브러리
from collections import deque # 유저 입력 저장
from dotenv import load_dotenv # 환경 변수 불러오기
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # 언어모델과 임베딩 불러오기
from langchain_community.vectorstores import FAISS # 백터 데이터 베이스 불러오기 위함
from langchain_core.prompts import ChatPromptTemplate # 프롬프트 생성
from langchain_core.runnables import RunnablePassthrough # 데이터를 그대로 전달


##############################################################

# 환경 변수 설정
load_dotenv() # .env 로드

api_key = os.getenv("OPENAI_API_KEY") # api 호출


# api 검증
if not api_key:
    raise EnvironmentError("Error: OpenAI_API_KEY is not set. Please configure it in your environment.")
os.environ["OpenAI_API_KEY"] = api_key

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


###############################################################


# 기본 위치 할당
db_path = "./food_db/" # 백터 데이터베이스 위치 
prompt_path = "./Prompts/" # 프롬프트 위치
json_path = "./log/" # json파일 위치 (로그 저장용)



###############################################################


# 기본 설정
model = ChatOpenAI(model="gpt-4o-mini") # 모델 불러오기
embeddings = OpenAIEmbeddings(model="text-embedding-3-small") # 임베딩 불러오기
recipes_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True) # 백터 db불러오기
retriever = recipes_store.as_retriever(search_type="similarity", search_kwargs={"k": 5}) # 리트리버 설정하기


###############################################################


# 프롬프트 불러오기
def load_prompts(path, system_files):
    """
    Prompts 폴더안의 여러개의 프롬프트를 하나로 불러와 합친다.
    각각의 txt 파일은 순서대로 합쳐지며 이는 각각의 기능을 대변한다.
    """
    system_message = [] # 빈 리스트를 생성
    
    for txt in system_files:
        try:
            # 위치에 저장되어 있는 파일 불러오기
            with open(os.path.join(path, txt), "r", encoding="UTF-8") as f:
                content = f.read().replace("\\n", "\n")
                system_message.append(("system", content))
        
        # 오류 처리
        except FileNotFoundError:
            logger.error(f"프롬프트 파일 '{txt}'이 존재하지 않습니다.")
            st.error(f"프롬프트 파일 '{txt}'을 찾을 수 없습니다.")
            st.stop()
        
        # 오류 처리 2
        except Exception as e:
            logger.error(f"프롬프트 파일 '{txt}' 읽기 실패: {e}")
            st.error(f"프롬프트 파일 '{txt}' 읽는 중 오류 발생: {e}")
            st.stop()
    
    # 리스트에 데이터 추가하기
    system_message.append(("user", "data : {data}\\n\\nQuestion: {question}"))
    return system_message


# 시스템 메세지로 프롬푸트 부르기
system_message = load_prompts(prompt_path, ["Require_decide.txt", "Food_recipe.txt", "Food_recommend.txt"])

# 프롬프트 생성하기
prompt = ChatPromptTemplate.from_messages(system_message)



###############################################################


# JSON 파일 생성하기

json_file = None  # 전역 변수로 초기화

# JSON 파일 생성 함수
def create_json_file(base_dir=json_path, prefix="output_log"):
    global json_file  # 전역 변수 사용
    if json_file is None:  # 파일이 없을 때만 생성
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        timestamp = time.strftime("%Y%m%d_%H") # 파일이름의 중복을 제거하기 위해 타임스탬프 사용
        json_file = os.path.join(base_dir, f"{prefix}_{timestamp}.json")
    return json_file



###############################################################


# JSON 파일에 기록 저장
# JSON 파일 저장 함수
def append_to_json(user_input, assistant_response):
    """
    유저 입력과 모델 응답을 JSON 파일에 추가합니다.
    """
    file_path = create_json_file()  # 항상 동일한 파일을 참조
    try:
        # 기존 JSON 데이터 로드
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        else:
            existing_data = []
        
        # 새로운 데이터를 기존 데이터에 추가
        new_record = {
            "user_input": user_input,
            "assistant_response": assistant_response
        }
        existing_data.append(new_record)

        # 데이터를 JSON 파일에 기록
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"JSON 저장 실패: {e}")
        
        
        
###############################################################

        
# TTS 음성 재생 함수
def play_audio(text):
    """
    gTTS를 이용해 음성을 생성하고 Streamlit에서 바로 재생.
    """
    tts = gTTS(text=text, lang="ko") # 언어 한글로 설정
    
    # 음성 파일을 메모리에 저장
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    
    # base64로 인코딩하여 Streamlit에서 재생 가능하도록 설정
    audio_base64 = base64.b64encode(audio_buffer.read()).decode()
    audio_html = f"""
        <audio autoplay controls>
            <source src="data:audio/mpeg;base64,{audio_base64}" type="audio/mpeg">
        </audio>
    """
    components.html(audio_html, height=80)  # 오디오 플레이어 삽입
        
        
        
###############################################################


# Debug PassThrough 설정
class DebugPassThrough(RunnablePassthrough):
    def invoke(self, *args, **kwargs):
        output = super().invoke(*args, **kwargs)
        return output


# ContextToText: 데이터 유실 방지
class ContextToText(RunnablePassthrough):
    def invoke(self, inputs, config=None, **kwargs):
        return {"data": inputs["data"], "question": inputs["question"]}


# 랭체인 연결
rag_chain_divide = {
    "data": retriever,
    "question": DebugPassThrough(),
} | DebugPassThrough() | ContextToText() | prompt | model



###############################################################


# Streamlit UI 구성
def initialize_session_state():
    """세션 상태 초기화"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = deque(maxlen=3)
    if "response" not in st.session_state:
        st.session_state.response = ""



###############################################################

# 실행 코드
def main():
    try:
        st.set_page_config(page_title="요리 전문가 챗봇", page_icon="🍳", layout="wide")
        st.title("요리 전문가 챗봇")
        st.write("질문을 입력하면 요리 관련 정보를 제공합니다.")

        # 세션 상태 초기화
        initialize_session_state()
        
        # json파일 생성하기
        create_json_file()
        
        
        # 사이드바 구성
        with st.sidebar:
            st.header("설정")
            
            # TTS on/off 설정
            tts_enabled = st.checkbox("TTS (텍스트 음성 변환)", value=False)

            # 초기화 버튼
            if st.button("대화 기록 초기화", key="reset_button"):
                st.session_state.chat_history.clear()  # chat_history 초기화
                st.success("대화 기록이 초기화되었습니다.")  # 메시지 출력


        # 사용자 입력 처리
        if query := st.chat_input("질문을 입력하세요"):
            # 사용자 질문 저장
            user_input = {"role": "user", "content": query}
            st.session_state.chat_history.append(user_input)

            with st.chat_message("user"):
                st.write(query)

            with st.chat_message("assistant"):
                with st.spinner("답변을 생성하는 중..."):
                    try:
                        # 모델 호출 및 응답 처리
                        model_input = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history])
                        response = rag_chain_divide.invoke(model_input)
                        assistant_response = {"role": "assistant", "content": response.content}
                        st.session_state.chat_history.append(assistant_response)

                        # JSON에 기록
                        append_to_json(query, response.content)
                        
                        st.write(response.content)
                        
                        if tts_enabled:
                            play_audio(response.content)

                    except Exception as e:
                        error_message = f"응답 생성 중 오류 발생: {e}"
                        error_data = {"role": "assistant", "content": error_message}
                        st.session_state.chat_history.append(error_data)
                        append_to_json(error_data)
                        st.error(error_message)


    except Exception as e:
        logger.error(f"앱 실행 중 오류 발생: {e}")
        st.error(f"앱 실행 중 오류 발생: {e}")


if __name__ == "__main__":
    main()

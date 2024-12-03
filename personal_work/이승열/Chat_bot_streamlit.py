import streamlit as st
import logging
import os
import json
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 환경 변수 설정
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise EnvironmentError("Error: OpenAI_API_KEY is not set. Please configure it in your environment.")
os.environ["OpenAI_API_KEY"] = api_key

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




db_path = "./food_db/"
# 기본 설정
model = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
recipes_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
retriever = recipes_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})


# 프롬프트 로드 함수
def load_prompts(path, system_files):
    system_message = []
    for txt in system_files:
        try:
            with open(os.path.join(path, txt), "r", encoding="UTF-8") as f:
                content = f.read().replace("\\n", "\n")
                system_message.append(("system", content))
                
        except FileNotFoundError:
            logger.error(f"프롬프트 파일 '{txt}'이 존재하지 않습니다.")
            st.error(f"프롬프트 파일 '{txt}'을 찾을 수 없습니다.")
            st.stop()
            
        except Exception as e:
            logger.error(f"프롬프트 파일 '{txt}' 읽기 실패: {e}")
            st.error(f"프롬프트 파일 '{txt}' 읽는 중 오류 발생: {e}")
            st.stop()
            
    system_message.append(("user", "data : {data}\\n\\nQuestion: {question}"))
    return system_message


# 프롬프트 경로
prompt_path = "./Prompts/"
system_message = load_prompts(prompt_path, ["Require_decide.txt", "Food_recipe.txt", "Food_recommend.txt"])
prompt = ChatPromptTemplate.from_messages(system_message)




# JSON 파일 설정
json_path = "./log/"
json_file = None  # 전역 변수로 초기화


# JSON 파일 생성 함수
def create_json_file(base_dir=json_path, prefix="output_log"):
    global json_file  # 전역 변수 사용
    if json_file is None:  # 파일이 없을 때만 생성
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        timestamp = time.strftime("%Y%m%d_%H")
        json_file = os.path.join(base_dir, f"{prefix}_{timestamp}.json")
    return json_file



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
        with open(file_path, "a", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"JSON 저장 실패: {e}")
        
        

# Debug PassThrough 설정
class DebugPassThrough(RunnablePassthrough):
    def invoke(self, *args, **kwargs):
        output = super().invoke(*args, **kwargs)
        return output

# ContextToText: 데이터 유실 방지
class ContextToText(RunnablePassthrough):
    def invoke(self, inputs, config=None, **kwargs):
        # inputs["data"] = inputs["data"][-3:]  # 마지막 3개의 데이터만 포함

        # 데이터를 그대로 전달
        return {"data": inputs["data"], "question": inputs["question"]}


# 랭체인 연결
rag_chain_divide = {
    "data": retriever,
    "question": DebugPassThrough(),
} | DebugPassThrough() | ContextToText() | prompt | model

        
        

# Streamlit UI 구성
def initialize_session_state():
    """세션 상태 초기화"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "response" not in st.session_state:
        st.session_state.response = ""



def main():
    try:
        st.set_page_config(page_title="요리 전문가 챗봇", page_icon="🍳", layout="wide")
        st.title("요리 전문가 챗봇")
        st.write("질문을 입력하면 요리 관련 정보를 제공합니다.")

        # 세션 상태 초기화
        initialize_session_state()
        create_json_file()

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

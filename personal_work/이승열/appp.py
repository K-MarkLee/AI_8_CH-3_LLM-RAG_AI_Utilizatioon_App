import os
import json
import time
import logging
from collections import deque
from uuid import uuid4
from datetime import datetime
from dotenv import load_dotenv

import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

######################################################################
# 환경 변수 설정
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("Error: OpenAI_API_KEY is not set. Please configure it in your environment.")
os.environ["OpenAI_API_KEY"] = api_key

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



######################################################################



# 모델 및 데이터 설정
model = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 벡터 데이터베이스 로드
VECTOR_PATH = "food_db/"
recipes = FAISS.load_local(VECTOR_PATH, embeddings, allow_dangerous_deserialization=True)

# 리트리버 설정
retriever = recipes.as_retriever(search_type="similarity", search_kwargs={"k": 5})


######################################################################

# 프롬프트 로드 함수
def load_prompts(path, system_files):
    system_message = []
    for txt in system_files:
        try:
            with open(os.path.join(path, txt), "r", encoding="UTF-8") as f:
                content = f.read().replace("\\n", "\n")
                system_message.append(("system", content))
        except FileNotFoundError:
            print(f"Warning: Prompt file '{txt}' not found.")
        except Exception as e:
            print(f"Error loading prompt '{txt}': {e}")
    system_message.append(("user", "data : {data}\\n\\nQuestion: {question}"))
    return system_message

system_message = load_prompts("Prompts/", ["Require_decide.txt", "Food_recipe.txt", "Food_recommend.txt"])
prompt = ChatPromptTemplate.from_messages(system_message)
print(prompt)


######################################################################



# 랭체인 연결
class DebugPassThrough(RunnablePassthrough):
    def invoke(self, *args, **kwargs):
        output = super().invoke(*args, **kwargs)
        return output

class ContextToText(RunnablePassthrough):
    def invoke(self, inputs, config=None, **kwargs):
        inputs["data"] = inputs["data"][-3:]  # 마지막 3개의 데이터만 포함
        return {"data": inputs["data"], "question": inputs["question"]}

rag_chain_divide = {
    "data": retriever,
    "question": DebugPassThrough(),
} | DebugPassThrough() | ContextToText() | prompt | model




# 대화 히스토리 및 JSON 저장 설정
chat_history = deque(maxlen=10)
log = []



def create_json_file(base_dir="personal_work/이승열/log", prefix="output_log"):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"{prefix}_{timestamp}.json")


output_file = create_json_file()



def save_to_json(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)



def get_response(user_input):
    chat_history.append({"role": "user", "content": user_input})
    model_input = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    response = rag_chain_divide.invoke(model_input)
    chat_history.append({"role": "assistant", "content": response.content})
    return response.content


######################################################################
# Streamlit UI
def main():
    st.set_page_config(
        page_title="요리 도우미",
        page_icon="🍳",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("요리 도우미 🍳")
    st.subheader("요리에 대한 질문에 답변을 제공합니다!")

    # 대화 인터페이스
    chat_container = st.container()
    with chat_container:
        for message in chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # 사용자 입력
    if query := st.chat_input("질문을 입력하세요"):
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("답변을 생성하는 중..."):
                try:
                    response = get_response(query)
                    st.write(response)
                    log.append({"질문": query, "답변": response, "기록": list(chat_history)})
                except Exception as e:
                    st.error(f"응답 생성 중 오류가 발생했습니다: {str(e)}")
    
    # 로그 저장 옵션
    if st.button("로그 저장"):
        save_to_json(output_file, log)
        st.success("로그가 저장되었습니다!")

if __name__ == "__main__":
    main()

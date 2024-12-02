# 패키지 파일
import os
import faiss
import json
import time
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain.schema import Document
from uuid import uuid4
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from collections import deque


######################################################################

load_dotenv()

# 환경 변수 설정
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise EnvironmentError("Error: OpenAI_API_KEY is not set. Please configure it in your environment.")
os.environ["OpenAI_API_KEY"] = api_key


model = ChatOpenAI(model ="gpt-4o-mini")



######################################################################


# 임베딩 모델 불러오기
embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")




######################################################################

# 백터 데이터 베이스 불러오기
recipes = FAISS.load_local("food_db/" ,embeddings, allow_dangerous_deserialization=True)




#####################################################################


# 리트리버 연결하기 (가장 유사한 문서 5개 추출하기)
retriever = recipes.as_retriever(search_type="similarity", search_kwargs={"k": 5})




#####################################################################


# # 위치 설정
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

# 호출
system_message = load_prompts("Prompts/", ["Require_decide.txt", "Food_recipe.txt", "Food_recommend.txt"])
prompt = ChatPromptTemplate.from_messages(system_message)





#####################################################################

# 랭체인 연결하기 (DebugPassthrough 를 통해서 유저의 인풋 잘 받나 확인)
class DebugPassThrough(RunnablePassthrough):
    def invoke(self, *args, **kwargs):
        output = super().invoke(*args,**kwargs)
        return output
    
    
class ContextToText(RunnablePassthrough):
    def invoke(self, inputs, config = None, **kwargs):
        
        #마지막 데이터를 넣음으로 업데이트
        inputs["data"] = inputs["data"][-3:]
        return {"data": inputs["data"], "question": inputs["question"]}
    

    
# 랭체인 연결
rag_chain_divide = {
    "data": retriever,  # 데이터베이스에서 검색된 데이터
    "question": DebugPassThrough(),  # 유저 입력
} | DebugPassThrough() | ContextToText()| prompt | model



#####################################################################

# json파일 저장

def create_json_file(base_dir="personal_work/이승열/log", prefix="output_log"):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"{prefix}_{timestamp}.json")


# JSON 파일에 기록 저장
def save_to_json(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        



#####################################################################


# streamlit의 사용을 위한 호출
def get_response(user_input):
    # 히스토리에 사용자 입력 추가
    chat_history.append({"role": "user", "content" : user_input})
    
    
    #모델의 입력 구성
    model_input = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    
    return rag_chain_divide.invoke(model_input)


#####################################################################


# LLM 시동
# 대화 히스토리를 저장할 공간
chat_history = deque(maxlen = 10)
log = []
output_file = create_json_file()


# while True:
#     print("-----------------------------")
    
    
#     query = input("질문을 입력해 주세요 (break 입력시 종료됩니다) : ")
    
    
#     if query.lower() == "break":
#         save_to_json(output_file,log)
#         break
    
    
#     # 히스토리에 사용자 입력 추가
#     chat_history.append({"role": "user", "content" : query})
    
    
#     #모델의 입력 구성
#     model_input = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
#     print(model_input)


#     #응답 생성
#     response = rag_chain_divide.invoke(model_input)
    
    
#     # 히스토리에 모델의 응답 추가
#     chat_history.append({"role": "assistant", "content": response.content})
    
    
#     #저장
#     record = {
#         "질문" : query,
#         "답변" : response.content,
#         "기록" : list(chat_history)
#     }
#     log.append(record)
    

#     print("Question : ", query)
#     print(response.content)
    
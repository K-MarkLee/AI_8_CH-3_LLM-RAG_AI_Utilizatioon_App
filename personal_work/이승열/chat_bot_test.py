# 패키지 파일
import os
import faiss
import json
import time

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

# 환경 변수 설정
os.environ["OpenAI_API_KEY"] = os.getenv("GPT_API")

model = ChatOpenAI(model ="gpt-4o")



######################################################################


# 임베딩 모델 불러오기
embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")




######################################################################

# 백터 데이터 베이스 불러오기
recipes = FAISS.load_local("./food_db." ,embeddings, allow_dangerous_deserialization=True)




#####################################################################


# 리트리버 연결하기 (가장 유사한 문서 5개 추출하기)
retriever = recipes.as_retriever(search_type="similarity", search_kwargs={"k": 5})




#####################################################################

# # 프롬프트 설정

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "너는 주어진 데이터로만 답변을 할 수 있어."),
#     ("system", "너는 요리의 전문가야."),
    
    
#     ("system", "질문에 답변할 때는 항상 데이터를 세밀히 학습하고 정확한 답변을 생성해야 해."),
#     ("system", "만약 데이터 안에 유저의 질문이 없다면, 양해를 구하고 관련 데이터를 제공할 수 없음을 알리도록 해."),
#     ("system", "질문에 계산이 필요한 경우, 데이터를 기반으로 정확하게 계산하여 결과를 제공해야 합니다."),
#     ("system", "질문에 대한 답변을 생성 하기전에 검증을 마친 후에 생성해줘."),
#     ("system", "부족한 부분이 있다고 생각하면, 다른 데이터를 참조해줘"),

    
#     ("system", "다음은 답변 형식의 예시야: \n"
#                "user: 미역국을 만들고 싶어.\n"
#                "ai: 미역국의 재료는 ~입니다.\n"
#                "ai: 미역국을 만드는 순서는 다음과 같습니다:\n"
#                "ai: 첫 번째 ~~~, 두 번째 ~~~입니다.\n"),
    
    
#     ("system", "또 다른 예시를 들어줄게:\n"
#                "user: 내가 만들려고 하는 미역국의 칼로리는 얼마야?\n"
#                "ai: 미역국의 칼로리는 1인분당 ~kcal입니다. 이 레시피는 ~인분 기준이므로, 총 칼로리는 ~kcal입니다.\n"),
    
    
#     ("system", "추가 예시:\n"
#                "user: 5인분의 레시피로 수정해줘.\n"
#                "ai: 현재 미역국 레시피는 ~인분 기준입니다. 5인분으로 수정된 재료는 다음과 같습니다:\n"
#                "ai: ~~~.\n"
#                "이때도 순서를 전부 알려줘"),
    
    
#     ("system", "추가 예시:\n"
#                "user: 된장찌개 끓이는법을 알려줘.\n"
#                "ai: 된장찌개는 재료에 따라 나뉩니다. 현재 ~ 한 된장찌개 레시피가 있습니다 어떤 된장찌개 레시피를원하십니까?:\n"
#                ),
    
    
#     ("user", "다음과 같은 데이터를 학습해:\n{data}"),
#     ("user", "그리고 질문에 답해:\n{question}")
# ])

# # 위치 설정
path = "Prompts/"


system_files = ["Require_decide.txt","Food_recipe.txt","Food_recommend.txt"]


system_message = []
for txt in system_files :
    with open(os.path.join(path, txt),"r", encoding="UTF-8") as f:
       
        content = f.read().replace("\\n", "\n")
        system_message.append(("system", content))


system_message.append(("user", "data : {data}\\n\\nQuestion: {question}"))


prompt = ChatPromptTemplate.from_messages(system_message)




#####################################################################

# 랭체인 연결하기 (DebugPassthrough 를 통해서 유저의 인풋 잘 받나 확인)
class DebugPassThrough(RunnablePassthrough):
    def invoke(self, *args, **kwargs):
        output = super().invoke(*args,**kwargs)
        return output
    
    
class ContextToText(RunnablePassthrough):
    def invoke(self, inputs, config = None, **kwargs):
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
    return rag_chain_divide.invoke(user_input)


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
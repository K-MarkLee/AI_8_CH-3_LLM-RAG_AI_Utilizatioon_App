import streamlit as st
import logging
import os
import json
import time
import io

from datetime import datetime
from gtts import gTTS
import base64
import tempfile
import requests
from urllib.parse import urljoin

from gtts import gTTS
import streamlit.components.v1 as components
import base64
from collections import deque
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


<<<<<<< HEAD

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
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"JSON 저장 실패: {e}")
        
        
        
# TTS 음성 재생 함수
def play_audio(text):
    """
    gTTS를 이용해 음성을 생성하고 Streamlit에서 바로 재생.
    """
    tts = gTTS(text=text, lang="ko")
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
        st.session_state.chat_history = deque(maxlen=3)
    if "response" not in st.session_state:
        st.session_state.response = ""

=======
def autoplay_audio(audio_content, autoplay=True):
    """음성 재생을 위한 HTML 컴포넌트 생성"""
    b64 = base64.b64encode(audio_content).decode()
    md = f"""
        <audio {' autoplay' if autoplay else ''} controls>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    return st.markdown(md, unsafe_allow_html=True)

def text_to_speech(text, lang='ko'):
    """텍스트를 음성으로 변환"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts = gTTS(text=text, lang=lang)
            tts.save(fp.name)
            with open(fp.name, 'rb') as audio_file:
                audio_bytes = audio_file.read()
            os.unlink(fp.name)
            return audio_bytes
    except Exception as e:
        logger.error(f"음성 변환 오류: {e}")
        return None

def fetch_github_files(repo_path, folder_path):
    """GitHub 저장소에서 파일 목록을 가져오는 함수"""
    try:
        # GitHub API URL 구성
        api_url = f"https://api.github.com/repos/{repo_path}/contents/{folder_path}"
        response = requests.get(api_url)
        response.raise_for_status()
        
        files = []
        for item in response.json():
            if item['type'] == 'file' and item['name'].endswith('.json'):
                files.append({
                    'name': item['name'],
                    'download_url': item['download_url']
                })
        return True, files
    except Exception as e:
        logger.error(f"GitHub 파일 목록 가져오기 실패: {e}")
        return False, str(e)

def download_github_file(file_url):
    """GitHub에서 파일을 다운로드하는 함수"""
    try:
        response = requests.get(file_url)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.error(f"파일 다운로드 실패: {e}")
        return None

def process_github_files(repo_path="Dawol2205/chatbot_test", folder_path="food_DB"):
    """GitHub 저장소에서 JSON 파일들을 처리하는 함수"""
    success, files = fetch_github_files(repo_path, folder_path)
    if not success:
        return False, f"파일 목록 가져오기 실패: {files}"

    documents = []
    for file in files:
        try:
            content = download_github_file(file['download_url'])
            if content:
                # JSON 파싱
                data = json.loads(content)
                
                # Document 객체 생성
                doc = Document(
                    page_content=json.dumps(data, ensure_ascii=False, indent=2),
                    metadata={"source": file['name']}
                )
                documents.append(doc)
                
        except Exception as e:
            logger.error(f"파일 처리 실패 ({file['name']}): {e}")
            continue

    if not documents:
        return False, "처리된 문서가 없습니다."
    
    return True, documents

def initialize_session_state():
    """세션 상태 초기화"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "안녕하세요! 요리 도우미입니다. 어떤 요리에 대해 알고 싶으신가요?",
                "audio": None
            }
        ]
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "custom_prompt" not in st.session_state:
        st.session_state.custom_prompt = """
아래 정보를 기반으로 사용자의 질문에 답변해주세요:
{context}

사용자 질문: {question}
답변: 주어진 정보를 바탕으로 상세하게 답변하겠습니다.
"""
    if "voice_enabled" not in st.session_state:
        st.session_state.voice_enabled = True
>>>>>>> e1873d44ad42954c5a41b0a685ba2b4de61b82e6


<<<<<<< HEAD
=======
def get_text_chunks(documents):
    """텍스트를 청크로 분할"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100
    )
    return text_splitter.split_documents(documents)

def create_vector_store(documents):
    """벡터 저장소 생성"""
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    return FAISS.from_documents(documents=documents, embedding=embeddings)

def save_vectorstore_local(vectorstore, directory=VECTOR_PATH):
    """벡터 저장소를 로컬에 저장"""
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(directory, f"vectorstore_{timestamp}.pkl")
        
        with open(file_path, 'wb') as f:
            pickle.dump(vectorstore, f)
        
        return True, file_path
    except Exception as e:
        logger.error(f"로컬 저장 오류: {e}")
        return False, str(e)

def load_vectorstore_local(file_path):
    """로컬에서 벡터 저장소를 불러오기"""
    try:
        with open(file_path, 'rb') as f:
            vectorstore = pickle.load(f)
        return True, vectorstore
    except Exception as e:
        logger.error(f"로컬 로드 오류: {e}")
        return False, str(e)

def get_conversation_chain(vectorstore, openai_api_key, custom_prompt):
    """대화 체인 생성"""
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-4o-mini', temperature=0)
    
    PROMPT = PromptTemplate(
        template=custom_prompt,
        input_variables=["context", "question"]
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        ),
        combine_docs_chain_kwargs={"prompt": PROMPT},
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )

    return conversation_chain
>>>>>>> e1873d44ad42954c5a41b0a685ba2b4de61b82e6

def main():
    try:
        st.set_page_config(page_title="요리 전문가 챗봇", page_icon="🍳", layout="wide")
        st.title("요리 전문가 챗봇")
        st.write("질문을 입력하면 요리 관련 정보를 제공합니다.")

        # 세션 상태 초기화
        initialize_session_state()
        create_json_file()
        
        
        # 사이드바 구성
        with st.sidebar:
            st.header("설정")
            
<<<<<<< HEAD
            # TTS on/off 설정
            tts_enabled = st.checkbox("TTS (텍스트 음성 변환)", value=False)
=======
            # 음성 출력 토글
            st.session_state.voice_enabled = st.toggle("음성 출력 활성화", value=st.session_state.voice_enabled)
            
            # API 키 입력
            openai_api_key = st.text_input("OpenAI API Key", type="password")
            if not openai_api_key:
                st.info("OpenAI API 키를 입력해주세요.", icon="🔑")
>>>>>>> e1873d44ad42954c5a41b0a685ba2b4de61b82e6

            # 초기화 버튼
            if st.button("대화 기록 초기화", key="reset_button"):
                st.session_state.chat_history.clear()  # chat_history 초기화
                st.success("대화 기록이 초기화되었습니다.")  # 메시지 출력

<<<<<<< HEAD
        

        # 사용자 입력 처리
        if query := st.chat_input("질문을 입력하세요"):
            # 사용자 질문 저장
            user_input = {"role": "user", "content": query}
            st.session_state.chat_history.append(user_input)

            with st.chat_message("user"):
                st.write(query)

=======
            # GitHub 파일 처리 섹션
            st.header("GitHub 파일 처리")
            if st.button("GitHub에서 파일 가져오기"):
                if not validate_api_key(openai_api_key):
                    st.error("유효한 OpenAI API 키를 입력해주세요.")
                    st.stop()

                try:
                    with st.spinner("GitHub에서 파일을 처리하는 중..."):
                        success, result = process_github_files()
                        
                        if success:
                            # 문서 청크 생성
                            chunks = get_text_chunks(result)
                            
                            # 벡터 저장소 생성
                            vectorstore = create_vector_store(chunks)
                            
                            # 세션에 저장
                            st.session_state.vectorstore = vectorstore
                            st.session_state.conversation = get_conversation_chain(
                                vectorstore, 
                                openai_api_key,
                                st.session_state.custom_prompt
                            )
                            st.success("GitHub 파일 처리 완료!")
                        else:
                            st.error(f"GitHub 파일 처리 실패: {result}")

                except Exception as e:
                    st.error(f"파일 처리 중 오류 발생: {str(e)}")
                    logger.error(f"처리 오류: {e}")
            
            # 벡터 파일 저장 버튼
            save_button = st.button("벡터 저장")

            # 벡터 파일 로드 섹션
            st.header("벡터 파일 불러오기")
            vector_files = []
            if os.path.exists(VECTOR_PATH):
                vector_files = [f for f in os.listdir(VECTOR_PATH) if f.endswith('.pkl')]
            
            if vector_files:
                selected_file = st.selectbox("저장된 벡터 파일 선택", vector_files)
                load_button = st.button("벡터 불러오기")
            else:
                st.info("저장된 벡터 파일이 없습니다.")

        # 벡터 파일 불러오기
        if vector_files and load_button and selected_file:
            if not validate_api_key(openai_api_key):
                st.error("유효한 OpenAI API 키를 입력해주세요.")
                st.stop()

            try:
                with st.spinner("벡터 저장소를 불러오는 중..."):
                    file_path = os.path.join(VECTOR_PATH, selected_file)
                    success, result = load_vectorstore_local(file_path)
                    
                    if success:
                        st.session_state.vectorstore = result
                        st.session_state.conversation = get_conversation_chain(
                            result, 
                            openai_api_key,
                            st.session_state.custom_prompt
                        )
                        st.success("벡터 저장소를 성공적으로 불러왔습니다!")
                    else:
                        st.error(f"벡터 저장소 불러오기 실패: {result}")
                        
            except Exception as e:
                st.error(f"벡터 파일 불러오기 중 오류 발생: {e}")
                logger.error(f"로컬 로드 오류: {e}")

        # 벡터 저장소 로컬 저장
        if save_button:
            if not st.session_state.vectorstore:
                st.error("저장할 벡터 데이터가 없습니다. 먼저 JSON 파일을 처리해주세요.")
                st.stop()

            try:
                with st.spinner("벡터 저장소를 저장하는 중..."):
                    success, result = save_vectorstore_local(st.session_state.vectorstore)
                    if success:
                        st.success(f"벡터 저장소를 저장했습니다! (경로: {result})")
                    else:
                        st.error(f"저장 실패: {result}")

            except Exception as e:
                st.error(f"저장 중 오류 발생: {str(e)}")
                logger.error(f"저장 오류: {e}")

        # 채팅 인터페이스
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    # 어시스턴트 메시지에 대해 음성 컨트롤 추가
                    if message["role"] == "assistant" and st.session_state.voice_enabled:
                        if message.get("audio") is None and message["content"]:
                            # 음성이 아직 생성되지 않은 경우 생성
                            audio_bytes = text_to_speech(message["content"])
                            if audio_bytes:
                                message["audio"] = audio_bytes

                        if message.get("audio"):
                            # 음성 컨트롤 표시
                            cols = st.columns([1, 4])
                            with cols[0]:
                                if st.button("🔊 재생", key=f"play_message_{i}"):
                                    autoplay_audio(message["audio"])
                            with cols[1]:
                                # 오디오 플레이어 표시 (컨트롤 포함)
                                autoplay_audio(message["audio"], autoplay=False)

        # 사용자 입력 처리
        if query := st.chat_input("질문을 입력하세요"):
            st.session_state.messages.append({"role": "user", "content": query, "audio": None})
            
            with st.chat_message("user"):
                st.write(query)

            if not st.session_state.conversation:
                response = "죄송합니다. 먼저 JSON 파일을 업로드하고 처리하거나 저장된 벡터를 불러와주세요."
                st.warning(response)
                
                if st.session_state.voice_enabled:
                    audio_bytes = text_to_speech(response)
                else:
                    audio_bytes = None
                    
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "audio": audio_bytes
                })
                
                if audio_bytes:
                    autoplay_audio(audio_bytes)
                
                st.stop()

>>>>>>> e1873d44ad42954c5a41b0a685ba2b4de61b82e6
            with st.chat_message("assistant"):
                with st.spinner("답변을 생성하는 중..."):
                    try:
                        # 모델 호출 및 응답 처리
                        model_input = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history])
                        response = rag_chain_divide.invoke(model_input)
                        assistant_response = {"role": "assistant", "content": response.content}
                        st.session_state.chat_history.append(assistant_response)

<<<<<<< HEAD
                        # JSON에 기록
                        append_to_json(query, response.content)
                        
                        st.write(response.content)
                        
                        if tts_enabled:
                            play_audio(response.content)
=======
                        st.write(response)

                        # 음성 출력 처리
                        if st.session_state.voice_enabled:
                            audio_bytes = text_to_speech(response)
                            if audio_bytes:
                                autoplay_audio(audio_bytes)
                        else:
                            audio_bytes = None

                        if source_documents:
                            with st.expander("참고 문서"):
                                for i, doc in enumerate(source_documents[:3], 1):
                                    st.markdown(f"**참고 {i}:** {doc.metadata.get('source', '알 수 없는 출처')}")
                                    st.markdown(f"```\n{doc.page_content[:200]}...\n```")

                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "audio": audio_bytes
                        })
>>>>>>> e1873d44ad42954c5a41b0a685ba2b4de61b82e6

                    except Exception as e:
                        error_message = f"응답 생성 중 오류 발생: {e}"
                        error_data = {"role": "assistant", "content": error_message}
                        st.session_state.chat_history.append(error_data)
                        append_to_json(error_data)
                        st.error(error_message)
<<<<<<< HEAD

=======
                        
                        if st.session_state.voice_enabled:
                            audio_bytes = text_to_speech(error_message)
                        else:
                            audio_bytes = None
                            
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_message,
                            "audio": audio_bytes
                        })
                        
                        if audio_bytes:
                            autoplay_audio(audio_bytes)
                            
                        logger.error(f"응답 생성 오류: {e}")
>>>>>>> e1873d44ad42954c5a41b0a685ba2b4de61b82e6

    except Exception as e:
        logger.error(f"앱 실행 중 오류 발생: {e}")
        st.error(f"앱 실행 중 오류 발생: {e}")


if __name__ == "__main__":
    main()

import streamlit as st
import logging
import pickle
import json
import os
from datetime import datetime
from gtts import gTTS
import base64
import tempfile
import requests
from urllib.parse import urljoin

from langchain_openai import ChatOpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 벡터 저장소 경로
VECTOR_PATH = "vectorstore"

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

def validate_api_key(api_key):
    """OpenAI API 키 형식 검증"""
    return api_key and len(api_key) > 20

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

def main():
    try:
        # 페이지 설정
        st.set_page_config(
            page_title="요리 도우미",
            page_icon="🍳",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # 세션 상태 초기화
        initialize_session_state()

        st.title("요리 도우미 🍳")

        # 사이드바 설정
        with st.sidebar:
            st.header("설정")
            
            # 음성 출력 토글
            st.session_state.voice_enabled = st.toggle("음성 출력 활성화", value=st.session_state.voice_enabled)
            
            # API 키 입력
            openai_api_key = st.text_input("OpenAI API Key", type="password")
            if not openai_api_key:
                st.info("OpenAI API 키를 입력해주세요.", icon="🔑")

            # 프롬프트 템플릿 설정
            st.header("프롬프트 템플릿")
            custom_prompt = st.text_area("RAG 프롬프트", value=st.session_state.custom_prompt)
            if custom_prompt != st.session_state.custom_prompt:
                st.session_state.custom_prompt = custom_prompt

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

            with st.chat_message("assistant"):
                with st.spinner("답변을 생성하는 중..."):
                    try:
                        result = st.session_state.conversation({"question": query})
                        response = result['answer']
                        source_documents = result.get('source_documents', [])

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

                    except Exception as e:
                        error_message = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
                        st.error(error_message)
                        
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

    except Exception as e:
        logger.error(f"앱 실행 중 오류 발생: {e}")
        st.error("앱 실행 중 오류가 발생했습니다. 새로고침을 시도해주세요.")

if __name__ == '__main__':
    main()

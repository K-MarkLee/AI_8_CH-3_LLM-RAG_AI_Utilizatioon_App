import streamlit as st
import logging
import os
import pickle
from datetime import datetime
from gtts import gTTS
import base64
import tempfile
from langchain_openai import ChatOpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def autoplay_audio(audio_content, autoplay=True):
    """음성 재생을 위한 HTML 컴포넌트 생성 (1.5배속)"""
    b64 = base64.b64encode(audio_content).decode()
    md = f"""
        <audio {' autoplay' if autoplay else ''} controls>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        <script>
            document.addEventListener('DOMContentLoaded', function() {{
                const audioElements = document.getElementsByTagName('audio');
                for(let audio of audioElements) {{
                    audio.playbackRate = 1.5;
                }}
            }});
        </script>
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
    if "voice_enabled" not in st.session_state:
        st.session_state.voice_enabled = True

def validate_api_key(api_key):
    """OpenAI API 키 형식 검증"""
    return api_key and len(api_key) > 20

def get_conversation_chain(vectorstore, openai_api_key):
    """대화 체인 생성"""
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-4o-mini', temperature=0)
    
    PROMPT = PromptTemplate(
        template="""
        아래 정보를 기반으로 사용자의 질문에 답변해주세요:
        {context}

        사용자 질문: {question}
        답변: 주어진 정보를 바탕으로 상세하게 답변하겠습니다.
        """,
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
            layout="wide"
        )

        # 세션 상태 초기화
        initialize_session_state()

        st.title("요리 도우미 🍳")

        # 음성 출력 토글 및 API 키 입력을 헤더 아래에 배치
        col1, col2 = st.columns([1, 2])
        with col1:
            st.session_state.voice_enabled = st.toggle("음성 출력 활성화", value=st.session_state.voice_enabled)
        with col2:
            openai_api_key = st.text_input("OpenAI API Key", type="password")
            if not openai_api_key:
                st.info("OpenAI API 키를 입력해주세요.", icon="🔑")

        # 채팅 인터페이스
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    if message["role"] == "assistant" and st.session_state.voice_enabled:
                        if message.get("audio") is None and message["content"]:
                            audio_bytes = text_to_speech(message["content"])
                            if audio_bytes:
                                message["audio"] = audio_bytes

                        if message.get("audio"):
                            cols = st.columns([1, 4])
                            with cols[0]:
                                if st.button("🔊 재생", key=f"play_message_{i}"):
                                    autoplay_audio(message["audio"])
                            with cols[1]:
                                autoplay_audio(message["audio"], autoplay=False)

        # 사용자 입력 처리
        if query := st.chat_input("질문을 입력하세요"):
            st.session_state.messages.append({"role": "user", "content": query, "audio": None})
            
            with st.chat_message("user"):
                st.write(query)

            if not validate_api_key(openai_api_key):
                response = "OpenAI API 키를 입력해주세요."
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
                        # food_DB 폴더에서 벡터스토어 직접 로드
                        if not st.session_state.conversation:
                            vectorstore_path = "food_DB/vectorstore.pkl"
                            if os.path.exists(vectorstore_path):
                                with open(vectorstore_path, 'rb') as f:
                                    vectorstore = pickle.load(f)
                                st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)
                            else:
                                st.error("벡터 데이터를 찾을 수 없습니다.")
                                st.stop()

                        result = st.session_state.conversation({"question": query})
                        response = result['answer']
                        source_documents = result.get('source_documents', [])

                        st.write(response)

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
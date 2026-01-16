import os
import json
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

def get_model(model_key_name=None) -> BaseChatModel:
    """
    .env 파일을 읽어서 API 키를 확인하고 적절한 모델을 반환합니다.
    우선순위: OPENAI_API_KEY -> OLLAMA_API_KEY
    """
    # OLLAMA_API_KEY 확인
    ollama_api_key = os.getenv("OLLAMA_API_KEY")
    if ollama_api_key:
        # OLLAMA_MODEL_DICT에서 OLLAMA_MODEL 값을 읽어서 사용
        model_dict_str = os.getenv("OLLAMA_MODEL_DICT", "{}")
        try:
            model_dict = json.loads(model_dict_str)
            model_name = model_dict.get(model_key_name, "gemma2:latest")
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 기본값 사용
            model_name = "gemma2:latest"
        
        return ChatOllama(model=model_name, temperature=0.5)

    # OPENAI_API_KEY 확인
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        # OPENAI_MODEL_DICT에서 GPT_4_MINI 값을 읽어서 사용
        model_dict_str = os.getenv("OPENAI_MODEL_DICT", "{}")
        try:
            model_dict = json.loads(model_dict_str)
            model_name = model_dict.get(model_key_name, "gpt-4o-mini")
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 기본값 사용
            model_name = "gpt-4o-mini"
        
        return ChatOpenAI(model=model_name, temperature=0.5)
        
    # 둘 다 없으면 기본값으로 OpenAI 사용 (API 키는 환경변수에서 자동으로 읽음)
    return ChatOllama(model="gemma2:latest", temperature=0.5)
import os
import json
from typing import Optional
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

# 기본 모델 설정
DEFAULT_OLLAMA_MODEL = "qwen3-vl"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OLLAMA_TEMPERATURE = 0.0
DEFAULT_OPENAI_TEMPERATURE = 0.5


def _parse_model_dict(env_key: str, default_model: str) -> dict[str, str]:
    """환경변수에서 모델 딕셔너리 파싱"""
    model_dict_str = os.getenv(env_key, "{}")
    try:
        return json.loads(model_dict_str)
    except json.JSONDecodeError:
        return {}


def _get_model_name(
    model_dict: dict[str, str], model_key_name: Optional[str], default: str
) -> str:
    """모델 딕셔너리에서 모델명 추출"""
    if model_key_name:
        return model_dict.get(model_key_name, default)
    return default


def get_model(model_key_name: Optional[str] = None) -> BaseChatModel:
    """
    .env 파일을 읽어서 API 키를 확인하고 적절한 모델을 반환합니다.
    우선순위: OPENAI_API_KEY -> OLLAMA_API_KEY -> 기본값 (Ollama)
    
    Args:
        model_key_name: 모델 딕셔너리에서 사용할 키 이름 (선택사항)
    
    Returns:
        BaseChatModel 인스턴스
    """
    # OLLAMA_API_KEY 확인
    ollama_api_key = os.getenv("OLLAMA_API_KEY")
    if ollama_api_key:
        model_dict = _parse_model_dict("OLLAMA_MODEL_DICT", DEFAULT_OLLAMA_MODEL)
        model_name = _get_model_name(model_dict, model_key_name, DEFAULT_OLLAMA_MODEL)
        return ChatOllama(model=model_name, temperature=DEFAULT_OLLAMA_TEMPERATURE)

    # OPENAI_API_KEY 확인
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        model_dict = _parse_model_dict("OPENAI_MODEL_DICT", DEFAULT_OPENAI_MODEL)
        model_name = _get_model_name(model_dict, model_key_name, DEFAULT_OPENAI_MODEL)
        return ChatOpenAI(model=model_name, temperature=DEFAULT_OPENAI_TEMPERATURE)

    # 둘 다 없으면 기본값으로 Ollama 사용
    return ChatOllama(model=DEFAULT_OLLAMA_MODEL, temperature=DEFAULT_OLLAMA_TEMPERATURE)
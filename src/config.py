"""
애플리케이션 설정 상수
"""

# RAG 설정
RAG_COLLECTION_NAME = "rag_collection"
RAG_PERSIST_DIRECTORY = "./rag_collection"
RAG_RETRIEVER_K = 3

# 임베딩 모델 설정
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-nli"
EMBEDDING_DEVICE = "cpu"

# 프론트엔드 설정
SERVER_URL = "http://localhost:8000"
FRONTEND_API_TIMEOUT = 600

# 에이전트 설정
CHATBOT_MAX_HISTORY = 5
DEFAULT_THREAD_ID = "default"

# API 엔드포인트
ENDPOINT_CHATBOT = "/chatbot"
ENDPOINT_PROJECT_CHATBOT = "/probject_chatbot"

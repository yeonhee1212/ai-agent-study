"""
공통 설정 및 유틸리티 모듈
"""
from langgraph.checkpoint.memory import MemorySaver
from .llm_model import get_model

# 모델 인스턴스를 모듈 레벨에서 한 번만 생성
llm_model = get_model()

print(f"llm model : {str(llm_model)}")

# 체크포인터를 생성하여 상태 저장/복원
checkpointer = MemorySaver()

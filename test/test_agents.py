"""
get_agent 함수 간단 테스트
현재 .env 파일 설정에 따라 어떤 모델이 반환되는지 확인
"""
import sys
import os
from dotenv import load_dotenv

sys.path.append(".")

from src.agents.agents import get_model

# .env 파일 로드
load_dotenv()

def main():
    print("=" * 60)
    print("get_agent() 테스트")
    print("=" * 60)
    
    # 현재 환경변수 확인
    print("\n[환경변수 확인]")
    openai_key = os.getenv("OPENAI_API_KEY")
    ollama_key = os.getenv("OLLAMA_API_KEY")
    openai_model_dict = os.getenv("OPENAI_MODEL_DICT")
    ollama_model_dict = os.getenv("OLLAMA_MODEL_DICT")
    
    print(f"OPENAI_API_KEY: {'있음' if openai_key else '없음'}")
    if openai_model_dict:
        print(f"OPENAI_MODEL_DICT: {openai_model_dict}")
    print(f"OLLAMA_API_KEY: {'있음' if ollama_key else '없음'}")
    if ollama_model_dict:
        print(f"OLLAMA_MODEL_DICT: {ollama_model_dict}")
    
    # get_agent() 호출
    print("\n[get_agent() 호출]")
    try:
        agent = get_model("GEMMA2")
        print(f"✅ 성공! 반환된 모델 타입: {type(agent).__name__}")
        
        # 모델 정보 출력
        if hasattr(agent, 'model_name'):
            print(f"   모델명: {agent.model_name}")
        elif hasattr(agent, 'model'):
            print(f"   모델명: {agent.model}")
        
        if hasattr(agent, 'temperature'):
            print(f"   Temperature: {agent.temperature}")
        
        print("\n✅ get_agent() 함수가 정상적으로 작동합니다!")
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

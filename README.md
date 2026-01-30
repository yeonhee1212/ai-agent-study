# ai-agent-study

## 실행방법
1. .env파일 수정
2. uv설치
    curl -LsSf https://astral.sh/uv/0.7.19/install.sh | sh
3. uv sync
4. source .venv/bin/activate
5. 서버 실행
    python main_server.py
6. 다른 터미널에서 화면 실행
    streamlit run main_front.py

## ppt파일 임베딩
1. root디렉토리에 doc폴더 생성해서 doc폴더 안에 ppt복붙, ppt->pdf내보내기 한 파일 복붙
    - doc/a.ppt
    - doc/a.pdf
2. sudo apt-get update
    sudo apt-get install -y poppler-utils 
    pdf2image는 poppler-utils가 필요함 
3. 스크립트 실행
    python src/scripts/create_po_vector_db.py 

## Agent소개
1. chatbot 
    - 대화 agent

2. probject_chatbot
    - proobject21 ppt파일 RAG agent

3. tmaxsoft_agent  
    - 멀티 에이전트 라우팅 구조
    ```
    HumanMessage
        ↓
    LLM Router
        ├─ web_search_graph : 웹 검색 agent
        ├─ general_chat_graph : 대화 agent
        └─ ProObject Agent Graph : proobject rag agent
    ```

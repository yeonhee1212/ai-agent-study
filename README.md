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
    python main_front.py

## ppt 임베딩
1. pdf로 내보내기 해서 같은 경로에 생성
2. sudo apt-get update
    sudo apt-get install -y poppler-utils 
    pdf2image는 poppler-utils가 필요함 
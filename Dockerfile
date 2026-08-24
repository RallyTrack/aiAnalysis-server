# 코드가 3.10+ 문법(X | None)을 쓰고 requirements가 3.12 기준으로 검증됨
FROM python:3.12-slim

WORKDIR /app

# opencv 런타임 라이브러리 + 시스템 ffmpeg (재인코딩용)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

EXPOSE 8000

# 서버 실행 (8000번 포트)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

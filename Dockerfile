FROM python:3.9

# 2. 작업 폴더 설정
WORKDIR /app

# 3. 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 소스 코드 복사
COPY . .

# 5. 서버 실행 (8000번 포트)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
RallyTrack AI 분석서버 엔트리포인트

실행: uvicorn main:app --reload --port 8000
"""
from fastapi import FastAPI
from routers.analyze import router as analyze_router

app = FastAPI(
    title="RallyTrack AI Analysis Server",
    description="배드민턴 영상 분석 API (MediaPipe + ST-GCN)",
    version="1.0.0",
)

# 라우터 등록
app.include_router(analyze_router)


@app.get("/")
def read_root():
    return {"message": "Hello, RallyTrack AI Server!"}

@app.get("/health") # 생존 확인(Health Check) 코드
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
"""
RallyTrack AI 분석서버 엔트리포인트

실행: uvicorn main:app --reload --port 8000
"""
from fastapi import FastAPI
from routers.analysis_router import router as analysis_router

app = FastAPI(
    title="RallyTrack AI Analysis Server",
    description="배드민턴 영상 분석 API",
    version="2.0.0",
)

app.include_router(analysis_router)


@app.get("/")
def read_root():
    return {"message": "RallyTrack AI Server"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

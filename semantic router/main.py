from fastapi import FastAPI
from app.api.vapi_router import router as vapi_router
from app.core.config import settings

app = FastAPI(title="Vapi Custom LLM Server")

app.include_router(vapi_router)

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "0.1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

from fastapi import FastAPI
from interfaces.api import router as api_router
from brand_brain.config import DEBUG_MODE
import uvicorn

app = FastAPI(
    title="Brand Brain API",
    version="1.8.0",
    description="FastAPI adapter for Brand Brain v1.8 CLI intelligence."
)

# Include Adapter Routes
app.include_router(api_router)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/version")
def version_check():
    return {
        "name": "Brand Brain",
        "version": "1.8.0",
        "interface": "fastapi"
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=DEBUG_MODE)

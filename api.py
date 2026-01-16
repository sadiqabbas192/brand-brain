from fastapi import FastAPI
from interfaces.api import router as api_router
from brand_brain.config import DEBUG_MODE
import uvicorn
import logging

from middleware import request_lifecycle_middleware

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

app = FastAPI(
    title="Brand Brain API",
    version="1.8.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- Middleware ---
app.middleware("http")(request_lifecycle_middleware)

# Include Adapter Routes
app.include_router(api_router)

@app.get("/health", tags=["System Status"])
def health_check():
    return {"status": "ok"}

@app.get("/version", tags=["System Status"])
def version_check():
    return {
        "name": "Brand Brain",
        "version": "1.8.0",
        "interface": "fastapi"
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=DEBUG_MODE)

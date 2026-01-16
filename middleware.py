import time
import logging
import json
from fastapi import Request
from starlette.concurrency import iterate_in_threadpool

logger = logging.getLogger("brand_brain")

async def request_lifecycle_middleware(request: Request, call_next):
    """
    Global middleware to log request lifecycle details:
    - Method, Path
    - Status Code
    - Duration
    - Query Params (Redacted)
    - JSON Body (Best Effort, Safe)
    """
    start_time = time.perf_counter()
    
    # 1. Capture Request Body (Safe)
    request_body_json = None
    try:
        # We need to read the body to log it, but reading consumes the stream.
        # We read it, then create a new stream for the app to consume.
        body_bytes = await request.body()
        
        # Restore the body for the actual route handler
        async def receive():
            return {"type": "http.request", "body": body_bytes}
        request._receive = receive
        
        if body_bytes:
            try:
                request_body_json = json.loads(body_bytes)
            except json.JSONDecodeError:
                request_body_json = "<Non-JSON Body>"
                
    except Exception:
        # Never fail request due to logging body
        request_body_json = "<Error reading body>"

    # 2. Process Request
    try:
        response = await call_next(request)
    except Exception as e:
        # Log the error and re-raise (uvicorn/fastapi will handle 500)
        duration = time.perf_counter() - start_time
        logger.error(f"Request failed: {str(e)}")
        raise e

    # 3. Calculate Metadata
    duration = time.perf_counter() - start_time
    
    # query_params -> dict, redacted
    query_params = dict(request.query_params)
    if "api_key" in query_params:
        query_params["api_key"] = "[REDACTED]"
    
    # Extract Brand ID if present in body (Best Effort)
    brand_id = None
    if isinstance(request_body_json, dict):
        brand_id = request_body_json.get("brand_id")

    # 4. Construct Log Line
    # Required Format: [METHOD] PATH | status=XXX | duration=XS | brand_id=... | args=... | json=...
    log_line = (
        f"[{request.method}] {request.url.path} | "
        f"status={response.status_code} | "
        f"duration={duration:.4f}s | "
        f"brand_id={brand_id} | "
        f"args={query_params} | "
        f"json={request_body_json}"
    )
    
    # 5. Log at INFO level
    logger.info(log_line)
    
    return response

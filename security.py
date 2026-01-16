from fastapi import Security, HTTPException, status, Query
from fastapi.security import APIKeyHeader
from typing import Optional

# Configuration
API_KEY = "digitalf5"

# Define Security Schemes for Swagger UI
# We only expose the Header scheme in Swagger UI for simplicity
api_key_header_scheme = APIKeyHeader(name="x-api-key", auto_error=False, description="Enter `digitalf5` here.")

async def verify_api_key(
    key_from_header: Optional[str] = Security(api_key_header_scheme),
    # Hidden from Swagger UI but still functional if sent manually
    key_from_query: Optional[str] = Query(None, alias="api_key") 
):
    """
    Verifies the API key.
    - Checks 'x-api-key' header first.
    - Checks 'api_key' query parameter second.
    - Raises 401 if valid key is missing.
    """
    if key_from_header == API_KEY:
        return key_from_header
        
    if key_from_query == API_KEY:
        return key_from_query
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key"
    )

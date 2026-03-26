import os
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY = os.environ.get('API_KEY', 'default-key')
api_key_header = APIKeyHeader(name='X-API-Key')

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import time
from app.flows.manager import flow_manager

router = APIRouter()

class Message(BaseModel):
    role: str
    content: str

class VapiRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = False
    call: Optional[Dict[str, Any]] = None 

@router.post("/chat/completions")
async def vapi_chat_completion(request: VapiRequest):
    # 1. Identify call_id
    call_id = "default_session"
    if request.call and "id" in request.call:
        call_id = request.call["id"]
    
    # 2. Get Flow
    flow = flow_manager.get_or_create_flow(call_id)
    
    # 3. Get last user message
    if not request.messages:
         return JSONResponse({"content": "No messages received"})
         
    last_message = request.messages[-1]
    
    if last_message.role != "user":
        # Fallback empty
        return JSONResponse({
            "id": "error",
            "choices": [{"message": {"content": ""}, "finish_reason": "stop"}]
        })

    user_text = last_message.content

    if request.stream:
        # Use real streamer
        generator = flow.process_input_stream(user_text)
        return StreamingResponse(vapi_sse_generator(generator), media_type="text/event-stream")
    else:
        # Non-streaming
        response_text = await flow.process_input(user_text)
        return {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": int(time.time()),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }]
        }

async def vapi_sse_generator(generator):
    chunk_id = f"chatcmpl-{int(time.time())}"
    
    async for text_chunk in generator:
        if not text_chunk:
            continue
            
        data = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "llama-3.1-70b-versatile",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": text_chunk},
                    "finish_reason": None
                }
            ]
        }
        # JSON formatting for Vapi
        yield f"data: {json.dumps(data)}\n\n"

    # End of stream
    data_end = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "llama-3.1-70b-versatile",
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ]
    }
    yield f"data: {json.dumps(data_end)}\n\n"
    yield "data: [DONE]\n\n"

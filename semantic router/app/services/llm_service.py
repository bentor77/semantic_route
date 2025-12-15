import os
from groq import Groq
from app.core.config import settings
from app.services.router_service import RouterService
from typing import AsyncGenerator

class SmartLLMService:
    def __init__(self, router_service: RouterService):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.router = router_service
        self.model = settings.GROQ_MODEL

    async def get_response_stream(self, text: str, system_message: str, history: list = None, tools: list = None) -> AsyncGenerator[str, None]:
        """
        Yields chunks of text from Groq.
        """
        messages = [{"role": "system", "content": system_message}]
        if history:
            messages.extend(history)
        
        messages.append({"role": "user", "content": text})

        try:
            # Prepare args
            kwargs = {
                "messages": messages,
                "model": self.model,
                "temperature": 0.7,
                "stream": True,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            stream = self.client.chat.completions.create(**kwargs)
            
            for chunk in stream:
                # Simple handling: yield text content. 
                # If tool_calls are present, we might want to handle them differently in future.
                # For now, we focus on speech output.
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                # NOTE: If we want to support tool executions, we need to capture tool_calls here,
                # execute them, and feed back to the LLM (internal loop) OR pass them downstream.
        except Exception as e:
            print(f"Groq API Error: {e}")
            yield "Lo siento, tuve un problema procesando tu solicitud."

    # Keep non-streaming version just in case
    async def get_response(self, text: str, system_message: str, history: list = None) -> str:
        full_response = ""
        async for chunk in self.get_response_stream(text, system_message, history):
            full_response += chunk
        return full_response

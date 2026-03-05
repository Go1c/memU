"""
MemU FastAPI Server for Zeabur deployment.
Supports OpenRouter as LLM provider and PostgreSQL for persistent memory.

Required environment variables:
  OPENROUTER_API_KEY  - Your OpenRouter API key
  DATABASE_URL        - PostgreSQL connection string (optional, falls back to in-memory)

Optional environment variables:
  CHAT_MODEL          - Chat model ID (default: anthropic/claude-3.5-sonnet)
  EMBED_MODEL         - Embedding model ID (default: openai/text-embedding-3-small)
  PORT                - Server port (default: 8000)
"""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

from memu.app import MemoryService
from memu.app.settings import DatabaseConfig, MetadataStoreConfig, VectorIndexConfig

memory_service: MemoryService | None = None


def get_llm_profiles() -> dict:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")

    chat_model = os.getenv("CHAT_MODEL", "anthropic/claude-3.5-sonnet")
    embed_model = os.getenv("EMBED_MODEL", "openai/text-embedding-3-small")

    profile = {
        "provider": "openrouter",
        "client_backend": "httpx",
        "base_url": "https://openrouter.ai",
        "api_key": api_key,
        "chat_model": chat_model,
        "embed_model": embed_model,
    }
    return {"default": profile, "embedding": profile}


def get_database_config() -> DatabaseConfig:
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return DatabaseConfig(
            metadata_store=MetadataStoreConfig(
                provider="postgres",
                dsn=dsn,
                ddl_mode="create",
            ),
            vector_index=VectorIndexConfig(
                provider="pgvector",
                dsn=dsn,
            ),
        )
    return DatabaseConfig()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global memory_service
    db_mode = "PostgreSQL" if os.getenv("DATABASE_URL") else "in-memory (no DATABASE_URL set)"
    print(f"Initializing MemU with OpenRouter + {db_mode}...")
    memory_service = MemoryService(
        llm_profiles=get_llm_profiles(),
        database_config=get_database_config(),
    )
    print("MemU initialized successfully.")
    yield


app = FastAPI(
    title="MemU Assistant",
    description="AI Memory Service powered by MemU + OpenRouter",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    user_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    memories_used: int
    memories_stored: int


class MemorizeRequest(BaseModel):
    content: str
    user_id: str = "default"


class MemorizeResponse(BaseModel):
    status: str
    items_created: int
    categories: int


class RecallResponse(BaseModel):
    query: str
    memories_found: int
    memories: list[dict]


@app.get("/")
async def root():
    return {
        "service": "MemU Assistant",
        "status": "running",
        "endpoints": {
            "GET /health": "Health check",
            "POST /chat": "Chat with memory-aware AI",
            "POST /memorize": "Store information in memory",
            "GET /recall?query=xxx": "Query stored memories",
        },
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "memory_service": memory_service is not None,
        "database": "postgres" if os.getenv("DATABASE_URL") else "inmemory",
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not memory_service:
        raise HTTPException(status_code=503, detail="Memory service not initialized")
    try:
        retrieve_result = await memory_service.retrieve(
            queries=[{"role": "user", "content": request.message}]
        )
        memories = retrieve_result.get("items", [])
        memory_context = [
            m.get("summary", str(m)) if isinstance(m, dict) else str(m)
            for m in memories[:5]
        ]

        if memory_context:
            response_text = (
                f"Based on memory context for '{request.message}':\n\n"
                + "\n".join(f"- {ctx[:200]}" for ctx in memory_context)
            )
        else:
            response_text = f"Received: '{request.message}'. No relevant memories yet."

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(f"User ({request.user_id}): {request.message}")
            temp_file = f.name

        try:
            memorize_result = await memory_service.memorize(
                resource_url=temp_file, modality="text"
            )
            memories_stored = len(memorize_result.get("items", []))
        finally:
            os.unlink(temp_file)

        return ChatResponse(
            response=response_text,
            memories_used=len(memories),
            memories_stored=memories_stored,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memorize", response_model=MemorizeResponse)
async def memorize(request: MemorizeRequest):
    if not memory_service:
        raise HTTPException(status_code=503, detail="Memory service not initialized")
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(f"[User: {request.user_id}] {request.content}")
            temp_file = f.name
        try:
            result = await memory_service.memorize(resource_url=temp_file, modality="text")
            return MemorizeResponse(
                status="stored",
                items_created=len(result.get("items", [])),
                categories=len(result.get("categories", [])),
            )
        finally:
            os.unlink(temp_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recall", response_model=RecallResponse)
async def recall(query: str, limit: int = 5):
    if not memory_service:
        raise HTTPException(status_code=503, detail="Memory service not initialized")
    try:
        result = await memory_service.retrieve(
            queries=[{"role": "user", "content": query}]
        )
        items = result.get("items", [])[:limit]
        memories = [
            {"summary": m.get("summary", str(m)), "category": m.get("category", "unknown")}
            if isinstance(m, dict) else {"summary": str(m), "category": "unknown"}
            for m in items
        ]
        return RecallResponse(query=query, memories_found=len(memories), memories=memories)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# main.py
# TO INSTALL LIBRARIES:
# pip install fastapi uvicorn langchain-openai langchain-core langchain-community python-dotenv
# TO LAUNCH SERVER:
# uvicorn main:app --reload
# API INFO: http://127.0.0.1:8000/docs (or whatever port you will use)

import os
from typing import Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ----------------------------
# Prompt setup
# ----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a concise, helpful assistant. Keep answers direct unless the user asks for depth."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

llm = ChatOpenAI(model=OPENAI_MODEL)  # uses OPENAI_API_KEY from environment
chain = prompt | llm

# ----------------------------
# Memory store (per-session)
# ----------------------------
session_store: Dict[str, ChatMessageHistory] = {}

def get_history_from_config(cfg: Any) -> ChatMessageHistory:
    session_id = None
    if isinstance(cfg, dict):
        cfg_conf = cfg.get("configurable") if isinstance(cfg.get("configurable"), dict) else {}
        session_id = cfg_conf.get("session_id")
    if not session_id:
        session_id = "default"
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

conversation = RunnableWithMessageHistory(
    chain,
    get_session_history=get_history_from_config,
    input_messages_key="input",
    history_messages_key="history",
)

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="LangChain + OpenAI Chatbot with Memory")

class ChatRequest(BaseModel):
    session_id: str
    user_input: str

@app.post("/chat")
async def chat(req: ChatRequest):
    """Send a message and get the assistant's reply, with per-session memory."""
    result = await conversation.ainvoke(
        {"input": req.user_input},
        config={"configurable": {"session_id": req.session_id}}
    )
    if hasattr(result, "content"):
        reply = result.content
    elif isinstance(result, dict) and "output" in result:
        reply = result["output"]
    else:
        reply = str(result)
    return {"reply": reply}

@app.post("/reset")
async def reset(req: ChatRequest):
    """Reset memory for a given session_id."""
    session_store.pop(req.session_id, None)
    return {"status": "cleared", "session_id": req.session_id}

# ----------------------------
# CLI mode for testing
# ----------------------------
def run_cli():
    session_id = os.getenv("SESSION_ID", "local-cli")
    print(f"Model: {OPENAI_MODEL}\nSession: {session_id}\nType 'exit' to quit.\n")
    
    import asyncio

    async def loop():
        while True:
            try:
                user = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()  # newline
                break
            if not user or user.lower() in {"exit", "quit"}:
                break
            result = await conversation.ainvoke(
                {"input": user},
                config={"configurable": {"session_id": session_id}},
            )
            if hasattr(result, "content"):
                reply = result.content
            elif isinstance(result, dict) and "output" in result:
                reply = result["output"]
            else:
                reply = str(result)
            print(f"Bot: {reply}\n")
    
    try:
        asyncio.run(loop())
    except RuntimeError:
        # Jupyter or active loop
        import nest_asyncio
        nest_asyncio.apply()
        asyncio.get_event_loop().run_until_complete(loop())

if __name__ == "__main__":
    # Direct CLI testing
    run_cli()

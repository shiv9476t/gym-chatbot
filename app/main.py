import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from app.rag import build_rag_chain

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

chain = build_rag_chain()

# In-memory session store — { session_id: [HumanMessage, AIMessage, ...] }
sessions = {}

class ChatRequest(BaseModel):
    question: str
    session_id: str

class ChatResponse(BaseModel):
    answer: str
        
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("frontend/demo.html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # Get or create history for this session
    if request.session_id not in sessions:
        sessions[request.session_id] = []

    history = sessions[request.session_id]

    today = datetime.now().strftime("%A %d %B %Y")
    # Run the chain
    answer = chain.invoke({
        "question": request.question,
        "chat_history": history,
        "today" : today
    })
    
    # Save turn to history
    history.append(HumanMessage(content=request.question))
    history.append(AIMessage(content=answer))

    return ChatResponse(answer=answer)


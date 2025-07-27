from fastapi import FastAPI
from pydantic import BaseModel
from bertmodel import answer_question, context  

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(req: QuestionRequest):
    answer = answer_question(req.question, context)
    return {"answer": answer}

@app.get("/")
async def root():
    return {"message": "Chatbot is ready! Go to /docs to interact with it."}
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

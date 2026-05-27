from fastapi import APIRouter
from pydantic import BaseModel

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from openai import OpenAI
import os

router = APIRouter()

CHROMA_DIR = "chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_model
)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

class ChatRequest(BaseModel):
    question: str

@router.post("/chat")
def chat_with_pdf(data: ChatRequest):

    docs = vectorstore.similarity_search(data.question, k=4)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are CampusGPT, an AI academic assistant.

    Use the context below to answer the student's question.

    Context:
    {context}

    Question:
    {data.question}
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response.choices[0].message.content

    return {
        "question": data.question,
        "answer": answer
    }
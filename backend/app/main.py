from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI

from app.api.routes.upload import router as upload_router
from app.api.routes.chat import router as chat_router

app = FastAPI()

app.include_router(upload_router)
app.include_router(chat_router)

@app.get("/")
def home():
    return {"message": "CampusGPT Backend Running"}